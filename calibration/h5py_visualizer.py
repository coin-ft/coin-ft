import h5py
import matplotlib.pyplot as plt
import numpy as np

def visualize_h5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        # 1. Pull Metadata (Attributes)
        sensor_name = f.attrs.get('sensor_name', 'Unknown Sensor')
        timestamp = f.attrs.get('timestamp', 'N/A')
        
        # 2. Pull Datasets
        # We use [:] to load the data into memory as a NumPy array
        sensor_data = f['sensor_cal_data'][:]
        ati_data = f['ati_cal_FT'][:]
        
        print(f"File: {file_path}")
        print(f"Sensor: {sensor_name} | Recorded at: {timestamp}")
        print(f"Data Shapes: Sensor {sensor_data.shape}, ATI {ati_data.shape}")

        # 3. Plot 12-Channel Sensor Data
        plt.figure(figsize=(12, 5))
        for i in range(sensor_data.shape[1]):
            plt.plot(sensor_data[:, i], label=f'Ch {i+1}', alpha=0.7)
        plt.title(f"Synced Sensor Data - {sensor_name}")
        plt.xlabel("Synced Sample Count")
        plt.ylabel("Raw Counts (Tared)")
        plt.legend(ncol=4, fontsize='small')
        plt.grid(True, alpha=0.3)

        # 4. Plot 6-Axis Reference Data (ATI)
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        labels = ["Fx (N)", "Fy (N)", "Fz (N)", "Mx (Nm)", "My (Nm)", "Mz (Nm)"]
        
        for i, ax in enumerate(axes.flatten()):
            ax.plot(ati_data[:, i], color='blue')
            ax.set_title(labels[i])
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f"ATI Reference Data - Session {timestamp}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# --- RUN IT ---
# Replace with your actual filename
filename = 'data/CFT24_calibrationData_20251219_111755_train.h5' 
visualize_h5_data(filename)