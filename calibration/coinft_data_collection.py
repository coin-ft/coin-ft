import numpy as np
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader, DigitalSingleChannelReader
import serial
import serial.tools.list_ports
import time
import h5py
import threading
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

#########################
# 1. CONTROL PANEL      #
#########################
COM_NAME = 'COM6'           # Serial port for CoinFT
BAUD_RATE = 1000000
NI_DEVICE = "Dev1"
ATI_CHANNELS = f"{NI_DEVICE}/ai0:5"
SYNC_CHANNEL = f"{NI_DEVICE}/port0/line1"
ATI_RATE = 1000             #
ATI_TARE_IN_SECONDS = 2.0
SAMPLING_DURATION = 40      # How long to collect data in [s]
M_ARM = 0.0115              # distance between the reference sensor surface and the CoinFT surface in [m]
SENSOR_NAME = 'CFT24'
ID = 'test'  # Unique identifier for the calibration session

#########################
# 2. INITIALIZATION     #
#########################
ati_data_list = []
sync_daq_list = []
coinft_data_list = []
stop_daq = threading.Event()
stop_cft = threading.Event()

def force_close_serial(port_name):
    try:
        test_ser = serial.Serial(port_name)
        test_ser.close()
        return True
    except:
        return False

# 2a. ATI Taring
print("Taring ATI...")
with nidaqmx.Task() as tare_task:
    tare_task.ai_channels.add_ai_voltage_chan(ATI_CHANNELS)
    reader = AnalogMultiChannelReader(tare_task.in_stream)
    tare_samples = int(ATI_TARE_IN_SECONDS * ATI_RATE)
    tare_task.timing.cfg_samp_clk_timing(rate=ATI_RATE, samps_per_chan=tare_samples)
    tare_buffer = np.zeros((6, tare_samples))
    reader.read_many_sample(tare_buffer, number_of_samples_per_channel=tare_samples, timeout=10.0)
    ATI_TARE = np.mean(tare_buffer, axis=1)

    ATI_CAL_MAT = np.array([
        [-.0027, -.0130, .2050, -3.5459, -.1941, 3.5575],
        [-.1037, 4.1186, .1131, -2.0527, .1362, -2.0528],
        [6.4822, -.1984, 6.3557, -.3824, 6.3658, -.3498],
        [-.0028, 0.0501, -.1827, -.0143, 0.1869, -.0343],
        [0.2164, -.0049, -.1102, 0.0489, -.1035, -.0379],
        [0.0041, -.1109, 0.0061, -.1103, 0.0054, -.1105]
    ])

# 2b. Serial Setup
force_close_serial(COM_NAME)
try:
    ser = serial.Serial(COM_NAME, BAUD_RATE, timeout=0.1)
    ser.write(b'i'); time.sleep(0.1); ser.reset_input_buffer()
    ser.write(b'q'); time.sleep(0.01)
    p_raw = ser.read(1)
    if not p_raw: raise RuntimeError("PSoC not responding")
    packet_size = ord(p_raw) - 1            # This should be 25
    num_sensors = (packet_size - 1) // 2    # This should be 12
    print(f"PSoC Ready. {num_sensors} Channels. Packet size: {packet_size}")
except Exception as e:
    sys.exit(f"Serial Error: {e}")

#########################
# 3. DATA ACQUISITION   #
#########################
def read_daq():
    """Hardware-clock synced capture of ATI and Sync line."""
    with nidaqmx.Task() as ai_task, nidaqmx.Task() as di_task:
        ai_task.ai_channels.add_ai_voltage_chan(ATI_CHANNELS)
        di_task.di_channels.add_di_chan(SYNC_CHANNEL)
        ai_task.timing.cfg_samp_clk_timing(rate=ATI_RATE, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        di_task.timing.cfg_samp_clk_timing(rate=ATI_RATE, source=f"/{NI_DEVICE}/ai/SampleClock", 
                                           sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        ai_task.in_stream.input_buf_size = 20000
        di_task.in_stream.input_buf_size = 20000
        di_task.start(); ai_task.start()
        while not stop_daq.is_set():
            samps = ai_task.in_stream.avail_samp_per_chan
            if samps > 0:
                ati_data_list.append(np.array(ai_task.read(number_of_samples_per_channel=samps)).T)
                sync_daq_list.append(np.array(di_task.read(number_of_samples_per_channel=samps)))
            else:
                time.sleep(0.01)

def read_coinft():
    """Reads PSoC data using standard s/i protocol."""
    """ 's' starts the CoinFT, 'i' puts it idle"""
    ser.write(b's')
    while not stop_cft.is_set():
        if ser.read(1) == b'\x02': 
            data = ser.read(packet_size)
            if data and data[-1] == 3: 
                coinft_data_list.append([data[i] + 256*data[i+1] for i in range(0, packet_size-1, 2)])
            else:
                print("Packet Error (Bad end framing byte)")
    ser.write(b'i')

#########################
# 4. RUN DATA COLLECTION     #
#########################
print("BEGIN DATA COLLECTION...")
t_daq = threading.Thread(target=read_daq)
t_cft = threading.Thread(target=read_coinft)

try:
    # REPLICATE MATLAB TIMING: ATI Starts First
    t_daq.start()
    time.sleep(1.0) # 1-second leading buffer for ATI
    
    t_cft.start()
    time.sleep(SAMPLING_DURATION)
    
    # REPLICATE MATLAB TIMING: CoinFT Stops, then ATI Stops
    stop_cft.set()
    t_cft.join(timeout=2.0)
    time.sleep(1.0) # 1-second trailing buffer for ATI
    
    stop_daq.set()
    t_daq.join(timeout=2.0)
finally:
    # Guaranteed cleanup
    if ser.is_open:
        ser.write(b'i')
        ser.close()
    print("Acquisition Stopped.")

#########################
# 5. SYNC & CALIBRATION #
#########################
ati_raw = np.vstack(ati_data_list)
sync_daq = np.concatenate(sync_daq_list).flatten()
coinft_raw = np.array(coinft_data_list)

# Robust Sync Line Masking for Bit 1
# This version handles: 
# 1. Booleans (True/False)
# 2. Raw Line 1 values (0 or 2)
# 3. Standard binary integers (0 or 1)
sync_daq_masked = (sync_daq.astype(int) > 0).astype(int)

# Baseline Taring (Samples 10-50)
baseline = np.mean(coinft_raw[10:50, :], axis=0)
coinft_tared = coinft_raw - baseline

# ATI Processing
ati_FT = (ati_raw - ATI_TARE) @ ATI_CAL_MAT.T
ati_FT[:, 3] += ati_FT[:, 1] * M_ARM # Mx
ati_FT[:, 4] -= ati_FT[:, 0] * M_ARM # My

# 1. Detect all transitions
sync_psoc = np.zeros(len(coinft_raw))
for i in range(0, len(coinft_raw), 6): sync_psoc[i:i+3] = 1
trans_psoc = np.where(np.diff(sync_psoc) != 0)[0]
trans_daq = np.where(np.diff(sync_daq_masked) != 0)[0]

# the difference in transition counts should not be larger than 1
# print(f"Sync Stats -> PSoC: {len(trans_psoc)}, ATI: {len(trans_daq)}")

# 2. ROBUST ALIGNMENT: Pairing i-th transitions
# We skip the first and last few to ensure we are in the 'meat' of the overlapping data
num_synced = min(len(trans_psoc), len(trans_daq)) - 2
sensor_cal_data, ati_cal_FT = np.zeros((num_synced, num_sensors)), np.zeros((num_synced, 6))


for i in range(num_synced):
    # Match the i-th pulse window in both sensors perfectly
    sensor_cal_data[i, :] = np.mean(coinft_tared[trans_psoc[i+1]+1 : trans_psoc[i+2]+1, :], axis=0)
    ati_cal_FT[i, :] = np.mean(ati_FT[trans_daq[i+1]+1 : trans_daq[i+2]+1, :], axis=0)


# THEN throw away the first row of the processed data, since on the CoinFT side the first sample tends to have electrical noise
if len(ati_cal_FT) > 1:
    ati_cal_FT = ati_cal_FT[1:, :]           
    sensor_cal_data = sensor_cal_data[1:, :] 


# Least Squares (2nd Order)
X = np.hstack([sensor_cal_data, sensor_cal_data**2])
A_T, _, _, _ = np.linalg.lstsq(X, ati_cal_FT, rcond=None)
calibrated_FT = X @ A_T # A_T is 24 x 6


#########################
# 5a. VALIDATION        #
#########################
rmse = np.sqrt(np.mean((calibrated_FT - ati_cal_FT)**2, axis=0))
print("\n" + "="*30)
print("LEAST SQUARES FIT CALIBRATION ACCURACY (RMSE)")
print("="*30)
axis_labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
units = ["N", "N", "N", "Nm", "Nm", "Nm"]

for i in range(6):
    print(f"{axis_labels[i]}: {rmse[i]:.4f} {units[i]}")
print("-" * 30)
print(f"AVG: {np.mean(rmse):.4f}")
print("="*30 + "\n")
print("While validated on training data, this gives a rough idea on the quality of the sensor data. There should at least be a learnable signal.")

#########################
# 6. PLOTTING           #
#########################
# Create a sample count vector (0, 1, 2, ... N)
samples_cal = np.arange(len(calibrated_FT))

plt.figure("Raw History", figsize=(10, 5))
# For raw history, we just plot against the default index (sample count)
off_plot = coinft_raw - np.mean(coinft_raw[20:40, :], axis=0)
for i in range(num_sensors): 
    plt.plot(off_plot[:, i], label=f'Ch{i+1}')
plt.title("Capacitive Sensor Output (Raw Sample Count)")
plt.xlabel("Sample Count")
plt.legend(ncol=4)

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
labels = ["Fx [N]", "Fy [N]", "Fz [N]", "Mx [Nm]", "My [Nm]", "Mz [Nm]"]
for i, ax in enumerate(axes.flatten()):
    # Plot using samples_cal instead of time_cal
    ax.plot(samples_cal, calibrated_FT[:, i], label='Calibrated CoinFT', linestyle='--', color='red')
    ax.plot(samples_cal, ati_cal_FT[:, i], label='ATI Reference', color='blue', alpha=0.6)
    
    ax.set_title(labels[i])
    ax.set_xlabel("Sample count") # Updated label
    ax.legend(fontsize='small')

plt.tight_layout()
plt.show()

# 7. Save results in HDF5 format


script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(script_dir)

data_dir = os.path.join(project_root, 'data')

# Create it if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created directory: {data_dir}")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
h5_filename = f'{SENSOR_NAME}_calibrationData_{ts}_{ID}.h5'
full_path = os.path.join(data_dir, h5_filename)

print(full_path)

with h5py.File(full_path, 'w') as f:
    # A. Store numerical arrays as "Datasets"
    # 'gzip' compression makes files much smaller without losing precision
    f.create_dataset('A_T', data=A_T, compression="gzip")
    f.create_dataset('sensor_cal_data', data=sensor_cal_data, compression="gzip")
    f.create_dataset('ati_cal_FT', data=ati_cal_FT, compression="gzip")
    
    # B. Store experiment info as "Attributes" (Metadata)
    f.attrs['sensor_name'] = SENSOR_NAME
    f.attrs['timestamp'] = ts
    f.attrs['m_arm'] = M_ARM
    f.attrs['ati_rate'] = ATI_RATE

print(f"Saved: {h5_filename}")