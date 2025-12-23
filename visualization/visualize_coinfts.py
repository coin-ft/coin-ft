#!/usr/bin/env python3
"""
Filename: visualize_coinfts.py
Description: Real-time visualizer for 2x CoinFT sensors via UART.
             No saving, no NTP. Just live plots.
"""

import time
import struct
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import onnxruntime as ort
import serial
import copy

# =========================
# Configuration
# =========================
# UART Settings
PORT_NAME    = "/dev/tty.usbmodem179386601"   
BAUD_RATE    = 115200
READ_TIMEOUT = 0.1

# Sensor / Model Settings
MODEL_PATHS   = ['NFT5_MLP_5L_norm_L2.onnx', 'NFT4_MLP_5L_norm_L2.onnx']
NORM_PATHS    = ['NFT5_norm_constants.mat',  'NFT4_norm_constants.mat']
LABELS        = ['Left Sensor (NFT5)', 'Right Sensor (NFT4)']

# Data Processing
INITIAL_SAMPLES = 500    # Samples to collect for tare
IGNORED_SAMPLES = 10     # Ignore start transient
WINDOW_SIZE     = 10     # Moving average window size

# Plotting
PLOT_HISTORY    = 5.0    # Seconds of history to show
PLOT_INTERVAL   = 10     # Update plot every N packets (lower = smoother but more CPU)

# Constants
COINFT_CH     = 12
HEADER_BYTES  = b"\x00\x00"
HEADER_LEN    = 2
BODY_LEN      = (COINFT_CH * 2) * 2  # 12 uint16s * 2 sensors

# =========================
# Helpers
# =========================

def load_norms(norm_paths):
    """Load normalization constants from .mat files."""
    out = []
    for path in norm_paths:
        mat = scipy.io.loadmat(path)['norm_const'].ravel()[0]
        # Ensure we extract as flat arrays for easy broadcasting
        out.append({
            'mu_x':  mat['mu_x'].astype(np.float32).flatten(),
            'sd_x':  mat['sd_x'].astype(np.float32).flatten(),
            'mu_y':  mat['mu_y'].astype(np.float32).flatten(),
            'sd_y':  mat['sd_y'].astype(np.float32).flatten()
        })
    return out

def start_stream(ser):
    """Handshake to start streaming."""
    print("Resetting sensors...")
    ser.write(b'i')
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.write(b's')
    time.sleep(0.05)

def read_packet(ser):
    """Reads one frame: [Header (2)] + [Sensor1 (24)] + [Sensor2 (24)]."""
    # 1. Look for Header
    # Simple sliding window to find 0x00 0x00
    while True:
        b = ser.read(1)
        if not b: return None # Timeout
        if b == b'\x00':
            b2 = ser.read(1)
            if b2 == b'\x00':
                break # Found header!

    # 2. Read Body
    body = ser.read(BODY_LEN)
    if len(body) != BODY_LEN:
        return None

    # 3. Parse
    vals = struct.unpack('<' + 'H' * (COINFT_CH * 2), body)
    data1 = np.array(vals[:COINFT_CH], dtype=np.float64)
    data2 = np.array(vals[COINFT_CH:], dtype=np.float64)
    return data1, data2

# =========================
# Main Loop
# =========================

def main():
    # 1. Setup Models
    print(f"Loading models: {MODEL_PATHS}")
    if len(MODEL_PATHS) != 2: raise ValueError("Expects 2 models")
    sessions = [ort.InferenceSession(p) for p in MODEL_PATHS]
    norms    = load_norms(NORM_PATHS)

    # 2. Setup Serial
    print(f"Opening {PORT_NAME}...")
    try:
        ser = serial.Serial(PORT_NAME, BAUD_RATE, timeout=READ_TIMEOUT)
    except Exception as e:
        print(f"Error opening serial: {e}")
        return

    start_stream(ser)

    # 3. Tare (Offset Calibration)
    print(f"Taring... ({INITIAL_SAMPLES} samples)")
    buffer1, buffer2 = [], []
    for _ in range(INITIAL_SAMPLES):
        pkt = read_packet(ser)
        if pkt:
            buffer1.append(pkt[0])
            buffer2.append(pkt[1])
    
    offsets = [
        np.mean(np.array(buffer1)[IGNORED_SAMPLES:], axis=0),
        np.mean(np.array(buffer2)[IGNORED_SAMPLES:], axis=0)
    ]
    print("Tare complete.")

    # 4. Setup Plotting
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # Layout:
    # [Sensor 1 Force] [Sensor 2 Force]
    # [Sensor 1 Moment] [Sensor 2 Moment]
    
    lines = [[], []] # Stores [ (fx, fy, fz), (mx, my, mz) ] for each sensor
    
    # Plot Buffers (Time, Fx, Fy, Fz, Mx, My, Mz)
    plot_data = [
        {'t': [], 'f': [[],[],[]], 'm': [[],[],[]]}, # Sensor 0
        {'t': [], 'f': [[],[],[]], 'm': [[],[],[]]}  # Sensor 1
    ]

    for i in range(2):
        # Force Plot
        ax_f = axes[0][i]
        lfx, = ax_f.plot([], [], 'r-', label='Fx')
        lfy, = ax_f.plot([], [], 'g-', label='Fy')
        lfz, = ax_f.plot([], [], 'b-', label='Fz')
        ax_f.set_title(f"{LABELS[i]} - Force")
        ax_f.set_ylim(-10, 10) # Set reasonable initial limits
        ax_f.legend(loc='upper right')
        ax_f.grid(True)

        # Moment Plot
        ax_m = axes[1][i]
        lmx, = ax_m.plot([], [], 'r--', label='Mx')
        lmy, = ax_m.plot([], [], 'g--', label='My')
        lmz, = ax_m.plot([], [], 'b--', label='Mz')
        ax_m.set_title(f"{LABELS[i]} - Moment")
        ax_m.set_ylim(-0.5, 0.5)
        ax_m.legend(loc='upper right')
        ax_m.grid(True)
        
        lines[i] = [ (lfx, lfy, lfz), (lmx, lmy, lmz) ]

    # Moving Average Filters
    ma_queues = [[], []]

    print("Starting visualization... (Ctrl+C to stop)")
    start_time = time.time()
    packet_count = 0

    try:
        while True:
            pkt = read_packet(ser)
            if pkt is None: continue

            now = time.time() - start_time

            # Process both sensors
            for i, raw in enumerate(pkt):
                # 1. Offset
                raw_zeroed = raw - offsets[i]

                # 2. Normalize
                raw_norm = (raw_zeroed.astype(np.float32) - norms[i]['mu_x']) / norms[i]['sd_x']
                
                # 3. Inference
                input_name = sessions[i].get_inputs()[0].name
                # Reshape to (1, 12) for ONNX
                pred_norm = sessions[i].run(None, {input_name: raw_norm.reshape(1, 12)})[0].flatten()

                # 4. Denormalize
                ft_val = pred_norm * norms[i]['sd_y'] + norms[i]['mu_y']

                # 5. Moving Average
                ma_queues[i].append(ft_val)
                if len(ma_queues[i]) > WINDOW_SIZE:
                    ma_queues[i].pop(0)
                
                ft_avg = np.mean(ma_queues[i], axis=0)

                # 6. Store for Plotting
                p = plot_data[i]
                p['t'].append(now)
                for j in range(3): p['f'][j].append(ft_avg[j])     # Fx, Fy, Fz
                for j in range(3): p['m'][j].append(ft_avg[j+3])   # Mx, My, Mz

                # Trim old data
                while p['t'] and (p['t'][-1] - p['t'][0] > PLOT_HISTORY):
                    p['t'].pop(0)
                    for j in range(3): p['f'][j].pop(0)
                    for j in range(3): p['m'][j].pop(0)

            # Update Plot
            packet_count += 1
            if packet_count % PLOT_INTERVAL == 0:
                for i in range(2):
                    t = plot_data[i]['t']
                    # Forces
                    for j, line in enumerate(lines[i][0]):
                        line.set_data(t, plot_data[i]['f'][j])
                    axes[0][i].relim()
                    axes[0][i].autoscale_view(True, True, True)

                    # Moments
                    for j, line in enumerate(lines[i][1]):
                        line.set_data(t, plot_data[i]['m'][j])
                    axes[1][i].relim()
                    axes[1][i].autoscale_view(True, True, True)
                
                plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.write(b'i')
        ser.close()
        print("Closed.")

if __name__ == "__main__":
    main()