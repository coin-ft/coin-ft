#!/usr/bin/env python3
"""
Filename: visualize_coinfts.py
Description: Adaptive Real-time visualizer for CoinFT sensors via Serial/UART.
             Supports 1 or 2 sensors based on NUM_COINFTS.
"""

import time
import struct
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import serial
import json
import os

# =========================
# User Configuration
# =========================
NUM_COINFTS  = 2   # Currently supports only 1 or 2 CoinFTs. The number of CoinFTs can be easily expanded by modifying the code.

# UART Settings
PORT_NAME    = "/dev/tty.usbmodem179386601"   
BAUD_RATE    = 115200
READ_TIMEOUT = 0.1

# Directory Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, '..', 'hardware_configs')

# Sensor / Model Settings
# List ALL potential models here. The script will slice this list based on NUM_COINFTS.
ALL_MODEL_FILES = ['CFT24_MLP.onnx', 'CFT24_MLP.onnx']  
ALL_NORM_FILES  = ['CFT24_norm.json', 'CFT24_norm.json'] 
ALL_LABELS      = ['Left Sensor', 'Right Sensor']

# Apply Selection
MODEL_FILES = ALL_MODEL_FILES[:NUM_COINFTS]
NORM_FILES  = ALL_NORM_FILES[:NUM_COINFTS]
LABELS      = ALL_LABELS[:NUM_COINFTS]

# Data Processing
INITIAL_SAMPLES = 500    # Samples to collect for tare
IGNORED_SAMPLES = 10     # Ignore start transient
WINDOW_SIZE     = 10     # Moving average window size

# Plotting
PLOT_HISTORY    = 5.0    # Seconds of history to show
PLOT_INTERVAL   = 40     # Update plot every N packets

# Constants
COINFT_CH     = 12
BYTES_PER_SENSOR = COINFT_CH * 2 # 12 uint16s * 2 bytes
HEADER_BYTES  = b"\x00\x00"
HEADER_LEN    = 2
BODY_LEN      = BYTES_PER_SENSOR * NUM_COINFTS 

# =========================
# Helpers
# =========================

def load_norms(norm_filenames):
    """Load normalization constants from JSON files."""
    out = []
    for fname in norm_filenames:
        path = os.path.join(CONFIG_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find config file: {path}")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        out.append({
            'mu_x':  np.array(data['mu_x'], dtype=np.float32),
            'sd_x':  np.array(data['sd_x'], dtype=np.float32),
            'mu_y':  np.array(data['mu_y'], dtype=np.float32),
            'sd_y':  np.array(data['sd_y'], dtype=np.float32)
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
    """Reads one frame for N sensors."""
    # 1. Look for Header
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
    # Unpack all uint16s at once
    vals = struct.unpack('<' + 'H' * (COINFT_CH * NUM_COINFTS), body)
    
    # Split into list of arrays
    sensor_data_list = []
    for i in range(NUM_COINFTS):
        start = i * COINFT_CH
        end   = (i + 1) * COINFT_CH
        sensor_data_list.append(np.array(vals[start:end], dtype=np.float64))
        
    return sensor_data_list

# =========================
# Main Loop
# =========================

def main():
    # 1. Setup Models
    print(f"Configured for {NUM_COINFTS} sensors.")
    print(f"Reading configs from: {CONFIG_DIR}")
    
    model_paths = [os.path.join(CONFIG_DIR, f) for f in MODEL_FILES]
    for p in model_paths:
        if not os.path.exists(p): raise FileNotFoundError(f"Model not found: {p}")

    sessions = [ort.InferenceSession(p) for p in model_paths]
    norms    = load_norms(NORM_FILES)

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
    # Initialize buffers for N sensors
    tare_buffers = [[] for _ in range(NUM_COINFTS)]
    
    for _ in range(INITIAL_SAMPLES):
        pkt_list = read_packet(ser)
        if pkt_list:
            for i in range(NUM_COINFTS):
                tare_buffers[i].append(pkt_list[i])
    
    offsets = []
    for i in range(NUM_COINFTS):
        arr = np.array(tare_buffers[i])
        # Check if we got enough data
        if len(arr) <= IGNORED_SAMPLES:
            raise RuntimeError("Not enough data for tare. Check connection.")
        offsets.append(np.mean(arr[IGNORED_SAMPLES:], axis=0))
        
    print("Tare complete.")

    # 4. Setup Plotting
    plt.ion()
    # Always create 2x2 grid, but hide unused columns if N=1
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    lines = [None] * NUM_COINFTS
    plot_data = []

    # Initialize Plot Data Structures
    for i in range(NUM_COINFTS):
        plot_data.append({'t': [], 'f': [[],[],[]], 'm': [[],[],[]]})

        # Force Plot
        ax_f = axes[0][i]
        lfx, = ax_f.plot([], [], 'r-', label='Fx')
        lfy, = ax_f.plot([], [], 'g-', label='Fy')
        lfz, = ax_f.plot([], [], 'b-', label='Fz')
        ax_f.set_title(f"{LABELS[i]} - Force")
        ax_f.legend(loc='upper right')
        ax_f.grid(True)

        # Moment Plot
        ax_m = axes[1][i]
        lmx, = ax_m.plot([], [], 'r--', label='Mx')
        lmy, = ax_m.plot([], [], 'g--', label='My')
        lmz, = ax_m.plot([], [], 'b--', label='Mz')
        ax_m.set_title(f"{LABELS[i]} - Moment")
        ax_m.legend(loc='upper right')
        ax_m.grid(True)
        
        lines[i] = [ (lfx, lfy, lfz), (lmx, lmy, lmz) ]

    # Hide unused subplots if NUM_COINFTS < 2
    if NUM_COINFTS < 2:
        for r in range(2):
            for c in range(NUM_COINFTS, 2):
                axes[r][c].set_visible(False)

    # Moving Average Filters
    ma_queues = [[] for _ in range(NUM_COINFTS)]

    print("Starting visualization... (Ctrl+C to stop)")
    start_time = time.time()
    packet_count = 0

    try:
        while True:
            pkt_list = read_packet(ser)
            if pkt_list is None: continue

            now = time.time() - start_time

            # Process N sensors
            for i, raw in enumerate(pkt_list):
                # 1. Offset
                raw_zeroed = raw - offsets[i]

                # 2. Normalize
                raw_norm = (raw_zeroed.astype(np.float32) - norms[i]['mu_x']) / norms[i]['sd_x']
                
                # 3. Inference
                input_name = sessions[i].get_inputs()[0].name
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
                for j in range(3): p['f'][j].append(ft_avg[j])     
                for j in range(3): p['m'][j].append(ft_avg[j+3])   

                # Trim old data
                while p['t'] and (p['t'][-1] - p['t'][0] > PLOT_HISTORY):
                    p['t'].pop(0)
                    for j in range(3): p['f'][j].pop(0)
                    for j in range(3): p['m'][j].pop(0)

            # Update Plot
            packet_count += 1
            if packet_count % PLOT_INTERVAL == 0:
                for i in range(NUM_COINFTS):
                    t = plot_data[i]['t']
                    if not t: continue 

                    # --- UPDATE FORCES ---
                    all_f = []
                    for j, line in enumerate(lines[i][0]):
                        f_data = plot_data[i]['f'][j]
                        line.set_data(t, f_data)
                        all_f.extend(f_data)
                    
                    # Dynamic Scaling Force
                    if all_f:
                        f_min, f_max = min(all_f), max(all_f)
                        axes[0][i].set_ylim(f_min - 3.0, f_max + 3.0)
                        axes[0][i].relim()
                        axes[0][i].autoscale_view(scaley=False)

                    # --- UPDATE MOMENTS ---
                    all_m = []
                    for j, line in enumerate(lines[i][1]):
                        m_data = plot_data[i]['m'][j]
                        line.set_data(t, m_data)
                        all_m.extend(m_data)

                    # Dynamic Scaling Moment
                    if all_m:
                        m_min, m_max = min(all_m), max(all_m)
                        axes[1][i].set_ylim(m_min - 0.1, m_max + 0.1)
                        axes[1][i].relim()
                        axes[1][i].autoscale_view(scaley=False)
                
                plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.write(b'i')
        ser.close()
        print("Closed.")

if __name__ == "__main__":
    main()