############################
# Notes on the Code:    #
# 1. The code is designed to read data from an ATI sensor and a custom CoinFT sensor. #
# 2. It uses the NI DAQ to read from the ATI sensor and a serial
# 3. For a different reference sensor, please modify the code accordingly. But the CoinFT part may as well remain the same.

import os
import time
import json
import queue
import serial
import threading
import numpy as np
import pandas as pd
import onnxruntime as ort
import nidaqmx
import matplotlib.pyplot as plt
from nidaqmx.stream_readers import AnalogMultiChannelReader
from matplotlib.widgets import Button, CheckButtons
from matplotlib.animation import FuncAnimation
from datetime import datetime
from threading import Lock

#########################
#  Configuration        #
#########################

# DAQ Settings (ATI)
NI_DEVICE = "Dev1/ai0:5"
ATI_SAMPLE_RATE = 1000
NUM_CHANNELS_ATI = 6  # ATI always has 6 channels
ATI_CAL_MATRIX = np.array([
    [-.0027,    -.0130,     .2050,     -3.5459,    -.1941,     3.5575],
    [-.1037,     4.1186,     .1131,     -2.0527,     .1362,    -2.0528],
    [ 6.4822,    -.1984,     6.3557,     -.3824,      6.3658,    -.3498],
    [-.0028,     0.0501,    -.1827,     -.0143,      0.1869,    -.0343],
    [ 0.2164,    -.0049,    -.1102,      0.0489,     -.1035,    -.0379],
    [ 0.0041,    -.1109,     0.0061,     -.1103,      0.0054,    -.1105]
])
M_ARM = 0.0115

# Serial Settings (CoinFT)
COM_NAME = 'COM6'
BAUD_RATE = 1000000
START_BYTE = 2
END_BYTE = 3

# Plot & Processing Settings
PLOT_DURATION = 10.0      # Seconds of history to display
MOVING_AVG_WINDOW = 30    # Rolling average window size
MAX_QUEUE_SIZE = 10000

# File Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
MODEL_PATH = os.path.join(DATA_DIR, 'CFT24_MLP.onnx')
JSON_PATH = os.path.join(DATA_DIR, 'CFT24_norm.json')

#########################
#  Global State         #
#########################

# Flags & synchronization
stop_flag = False
ati_lock = Lock()

# Data Queues
ati_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
sensor_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

# Data Storage for Saving
all_data_records = []

# Latest ATI State (Thread-safe)
latest_ati = None
latest_ati_time = None

# Calibration / Offset State
offset_CoinFT = np.zeros(6) # Will be resized dynamically
offset_CoinFT_list = []
read_count = 0
got_initial_offset = False

#########################
#  Resource Loading     #
#########################

print(f"Loading model from: {MODEL_PATH}")
print(f"Loading constants from: {JSON_PATH}")

# 1. Load ONNX Model
ort_session = ort.InferenceSession(MODEL_PATH)

# 2. Load Normalization Constants
with open(JSON_PATH, 'r') as f:
    norm_data = json.load(f)

mu_x = np.array(norm_data['mu_x'])
sd_x = np.array(norm_data['sd_x'])
mu_y = np.array(norm_data['mu_y'])
sd_y = np.array(norm_data['sd_y'])

# 3. Perform ATI Taring
print("Taring ATI sensor...")
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(NI_DEVICE, min_val=-10.0, max_val=10.0)
    task.timing.cfg_samp_clk_timing(rate=ATI_SAMPLE_RATE, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
    reader = AnalogMultiChannelReader(task.in_stream)
    
    tare_samples = int(0.5 * ATI_SAMPLE_RATE)
    data_tare = np.zeros((NUM_CHANNELS_ATI, tare_samples))
    reader.read_many_sample(data_tare, tare_samples, timeout=10.0)
    ATI_TARE = data_tare.mean(axis=1)
print("ATI Taring complete.")

#########################
#  Serial Handshake     #
#########################

ser = serial.Serial(COM_NAME, BAUD_RATE, timeout=0.1)
try:
    ser.write(b'i')
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.write(b'q')
    time.sleep(0.01)
    
    packet_size_raw = ser.read(1)
    if len(packet_size_raw) < 1:
        raise RuntimeError("Failed to read packet size from sensor")
    
    packet_size_excludeStartByte = ord(packet_size_raw) - 1
    num_Channels = (packet_size_excludeStartByte - 1) // 2
    print(f"Detected Sensor Channels: {num_Channels}")

except Exception as e:
    ser.close()
    raise e

#########################
#  Buffer Init          #
#########################

t_ati_store = np.array([])
ati_data_store = np.empty((0, 6))

t_sens_store = np.array([])
sens_data_store = np.empty((0, 6))              # Calibrated is always 6
raw_sens_data_store = np.empty((0, num_Channels)) # Raw depends on sensor

#########################
#  Worker Functions     #
#########################

def read_ati():
    """Reads data from the ATI sensor via NI DAQ."""
    global stop_flag, latest_ati, latest_ati_time
    
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(NI_DEVICE, min_val=-10.0, max_val=10.0)
        task.timing.cfg_samp_clk_timing(rate=ATI_SAMPLE_RATE, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        reader = AnalogMultiChannelReader(task.in_stream)
        
        buffer_size = 5
        data = np.zeros((6, buffer_size))

        while not stop_flag:
            num_read = reader.read_many_sample(data, number_of_samples_per_channel=buffer_size, timeout=1.0)
            timestamp_base = time.time()
            
            # Calibration calculation
            volts = data.T
            ati_ft = (volts - ATI_TARE) @ ATI_CAL_MATRIX.T
            
            # Mechanical offsets
            ati_ft[:, 3] = ati_ft[:, 3] + ati_ft[:, 1] * M_ARM
            ati_ft[:, 4] = ati_ft[:, 4] - ati_ft[:, 0] * M_ARM

            for i in range(num_read):
                sample_time = timestamp_base + i/ATI_SAMPLE_RATE
                sample_data = ati_ft[i, :].copy()
                
                with ati_lock:
                    latest_ati = sample_data
                    latest_ati_time = sample_time
                
                ati_queue.put((sample_time, sample_data))

def read_CoinFT():
    """Reads data from the custom CoinFT sensor via Serial."""
    global stop_flag, read_count, got_initial_offset, offset_CoinFT, offset_CoinFT_list
    
    initial_sample_num = 1500
    ft_bias_ready = False
    ft_bias = None

    while not stop_flag:
        byte = ser.read(1)
        if not byte or byte[0] != START_BYTE:
            continue

        data = ser.read(packet_size_excludeStartByte)
        if len(data) < packet_size_excludeStartByte:
            continue

        if data[-1] == END_BYTE:
            # Parse binary data
            sensor_vals = []
            for byte_num in range(0, packet_size_excludeStartByte-1, 2):
                low = data[byte_num]
                high = data[byte_num+1]
                sensor_vals.append(low + 256*high)
            
            sensor_data = np.array(sensor_vals, dtype=np.float64)
            read_count += 1

            # Calibration Phase
            if read_count <= initial_sample_num:
                offset_CoinFT_list.append(sensor_data)
                continue
            elif read_count == initial_sample_num + 1:
                offset_CoinFT = np.mean(offset_CoinFT_list[5:], axis=0)
                got_initial_offset = True
                print(f"CoinFT Offset Calibrated.")
                continue

            if not got_initial_offset:
                continue

            # Inference Phase
            sensor_data_offsetted = sensor_data - offset_CoinFT
            
            # Prepare for ONNX
            x_norm = (sensor_data_offsetted - mu_x) / sd_x
            x_input = x_norm.astype(np.float32).reshape(1, -1)
            
            calibrated_ft = ort_session.run(None, {"input": x_input})[0].flatten()
            calibrated_ft = calibrated_ft * sd_y + mu_y

            # Bias correction (tare at start of inference)
            if not ft_bias_ready:
                ft_bias = calibrated_ft
                ft_bias_ready = True
            else:
                calibrated_ft = calibrated_ft - ft_bias

            timestamp = time.time()
            
            # Get synced ATI data
            with ati_lock:
                ati_snapshot = latest_ati.copy() if latest_ati is not None else [np.nan]*6

            # Store Record
            row = [timestamp]
            # 1. Raw Sensor Data
            row.extend(sensor_data_offsetted.tolist() if len(sensor_data_offsetted) == num_Channels else [np.nan]*num_Channels)
            # 2. ATI Data (Ground Truth)
            row.extend(ati_snapshot)
            # 3. Model Prediction
            row.extend(calibrated_ft.tolist() if len(calibrated_ft) == 6 else [np.nan]*6)
            
            all_data_records.append(row)

            sensor_queue.put((timestamp, calibrated_ft, sensor_data_offsetted))

#########################
#  Plotting Logic       #
#########################

def moving_average_2d(arr, w=5):
    if arr.shape[0] < w:
        return arr
    out = np.zeros_like(arr)
    kernel = np.ones(w) / w
    for col in range(arr.shape[1]):
        out[:, col] = np.convolve(arr[:, col], kernel, mode='same')
    return out

def update_plot(frame):
    global t_ati_store, ati_data_store
    global t_sens_store, sens_data_store, raw_sens_data_store

    now = time.time()
    cutoff = now - PLOT_DURATION

    # 1. Consume ATI Queue
    new_t_ati = []
    new_ati = []
    while not ati_queue.empty():
        ts, d = ati_queue.get()
        new_t_ati.append(ts)
        new_ati.append(d)
    
    if new_t_ati:
        t_ati_store = np.concatenate([t_ati_store, np.array(new_t_ati)])
        ati_data_store = np.vstack([ati_data_store, np.array(new_ati)])

    # 2. Consume Sensor Queue
    new_t_sens = []
    new_sens = []
    new_raw = []
    while not sensor_queue.empty():
        ts, cft, raw = sensor_queue.get()
        new_t_sens.append(ts)
        new_sens.append(cft)
        new_raw.append(raw)
    
    if new_t_sens:
        t_sens_store = np.concatenate([t_sens_store, np.array(new_t_sens)])
        sens_data_store = np.vstack([sens_data_store, np.array(new_sens)])
        # Check raw shape before stacking
        raw_arr = np.array(new_raw)
        if raw_arr.shape[1] == raw_sens_data_store.shape[1]:
            raw_sens_data_store = np.vstack([raw_sens_data_store, raw_arr])

    # 3. Prune Old Data
    if t_ati_store.size > 0:
        mask = t_ati_store >= cutoff
        t_ati_store = t_ati_store[mask]
        ati_data_store = ati_data_store[mask]

    if t_sens_store.size > 0:
        mask = t_sens_store >= cutoff
        t_sens_store = t_sens_store[mask]
        sens_data_store = sens_data_store[mask]
        raw_sens_data_store = raw_sens_data_store[mask]

    # 4. Prepare Plot Data
    t_plot = t_sens_store
    sens_plot_data = np.empty((0, 6))
    ati_plot_matched = np.empty((0, 6))

    if t_plot.size > 0:
        # Filter sensor data
        sens_plot_data = moving_average_2d(sens_data_store, w=MOVING_AVG_WINDOW)
        
        # Match ATI data to Sensor timestamps
        if t_ati_store.size > 0:
            indices = np.searchsorted(t_ati_store, t_plot, side='right') - 1
            indices = np.clip(indices, 0, t_ati_store.size - 1)
            ati_plot_matched = ati_data_store[indices, :]
        else:
            ati_plot_matched = np.zeros_like(sens_plot_data)

    # 5. Update Lines
    if t_plot.size > 0:
        rel_time = t_plot - t_plot[0]
        
        for idx, i in enumerate(force_indices):
            if visibility[i]:
                lines_ati_force[idx].set_data(rel_time, ati_plot_matched[:, i])
                lines_sens_force[idx].set_data(rel_time, sens_plot_data[:, i])
            else:
                lines_ati_force[idx].set_data([], [])
                lines_sens_force[idx].set_data([], [])

        for idx, i in enumerate(torque_indices):
            if visibility[i]:
                lines_ati_torque[idx].set_data(rel_time, ati_plot_matched[:, i])
                lines_sens_torque[idx].set_data(rel_time, sens_plot_data[:, i])
            else:
                lines_ati_torque[idx].set_data([], [])
                lines_sens_torque[idx].set_data([], [])

        ax_force.relim(visible_only=True)
        ax_force.autoscale_view()
        ax_torque.relim(visible_only=True)
        ax_torque.autoscale_view()

#########################
#  GUI Setup            #
#########################

fig, (ax_force, ax_torque) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, hspace=0.25)

lines_ati_force, lines_sens_force = [], []
lines_ati_torque, lines_sens_torque = [], []
channels_labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
force_indices = [0, 1, 2]
torque_indices = [3, 4, 5]
visibility = [True]*6
color_map = ['red', 'green', 'blue']

# Initialize Force Axes
for idx, i in enumerate(force_indices):
    l_ati, = ax_force.plot([], [], label=f'ATI {channels_labels[i]}', linestyle='-', color=color_map[idx], linewidth=2)
    l_sens, = ax_force.plot([], [], label=f'Sens {channels_labels[i]}', linestyle=':', color=color_map[idx], linewidth=2)
    lines_ati_force.append(l_ati)
    lines_sens_force.append(l_sens)
ax_force.set_ylabel('Force (N)')
ax_force.legend(loc='upper left')
ax_force.set_title('Forces')
ax_force.grid(True)

# Initialize Torque Axes
for idx, i in enumerate(torque_indices):
    l_ati, = ax_torque.plot([], [], label=f'ATI {channels_labels[i]}', linestyle='-', color=color_map[idx], linewidth=2)
    l_sens, = ax_torque.plot([], [], label=f'Sens {channels_labels[i]}', linestyle=':', color=color_map[idx], linewidth=2)
    lines_ati_torque.append(l_ati)
    lines_sens_torque.append(l_sens)
ax_torque.set_xlabel('Time (s)')
ax_torque.set_ylabel('Torque (Nm)')
ax_torque.legend(loc='upper left')
ax_torque.set_title('Torques')
ax_torque.grid(True)

# GUI Controls
check_ax = plt.axes([0.01, 0.4, 0.08, 0.2])
check = CheckButtons(check_ax, channels_labels, visibility)

def toggle_visibility(label):
    i = channels_labels.index(label)
    visibility[i] = not visibility[i]
check.on_clicked(toggle_visibility)

stop_ax = plt.axes([0.85, 0.02, 0.1, 0.05])
stop_button = Button(stop_ax, 'Stop')
def stop(event):
    global stop_flag
    stop_flag = True
stop_button.on_clicked(stop)

def on_close(event):
    global stop_flag
    stop_flag = True
fig.canvas.mpl_connect('close_event', on_close)

#########################
#  Main Execution       #
#########################

if __name__ == "__main__":
    # 1. Start Serial Streaming
    ser.write(b's')
    
    # 2. User Prompt
    print("\n" + "="*50)
    print(" >>> SYSTEM READY.")
    print(" >>> PRESS [ENTER] TO START DATA COLLECTION & PLOTTING")
    print("="*50 + "\n")
    input()

    # 3. Start Threads
    ati_thread = threading.Thread(target=read_ati, daemon=True)
    sens_thread = threading.Thread(target=read_CoinFT, daemon=True)
    ati_thread.start()
    sens_thread.start()

    # 4. Start Animation Loop
    ani = FuncAnimation(fig, update_plot, interval=50, blit=False, cache_frame_data=False)
    plt.show()

    # 5. Cleanup
    print("Stopping threads...")
    stop_flag = True
    ati_thread.join()
    sens_thread.join()
    ser.close()

    # 6. Save Data
    print("Saving data...")
    
    # Dynamic Headers
    raw_cols = [f'CoinFT_offset_{i+1}' for i in range(num_Channels)]
    ati_cols = [f'ATI_{c}' for c in channels_labels]
    coinft_calib_cols = [f'CoinFT_calib_{c}' for c in channels_labels]
    
    columns = ['Time'] + raw_cols + ati_cols + coinft_calib_cols
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'CoinFT_data_{timestamp_str}.csv'
    
    # Ensure the directory exists (create it if missing)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created missing directory: {DATA_DIR}")

    # Combine the data directory with the filename
    full_save_path = os.path.join(DATA_DIR, filename)
    # -----------------------------
    
    if all_data_records:
        df = pd.DataFrame(all_data_records, columns=columns)
        df.to_csv(full_save_path, index=False)
        print(f"Saved {len(all_data_records)} records to {full_save_path}")
    else:
        print("No data recorded.")