import os
import glob
import numpy as np
import h5py
import json

# --- CONFIGURATION ---
SENSOR_NAME = 'CFT24'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to raw data (relative to this script)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
# Path to save processed data
SAVE_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
JSON_FILENAME = f'{SENSOR_NAME}_norm.json'  # Name of your JSON constants file

def process_data():
    # 1. Setup Directories
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    # 2. Find all HDF5 files matching the sensor name
    search_path = os.path.join(DATA_DIR, f"*{SENSOR_NAME}_calibrationData_*.h5")
    file_list = glob.glob(search_path)
    
    if not file_list:
        print(f"No files found in {search_path}")
        return

    print(f"Found {len(file_list)} files. Sorting by filename suffix...")

    # 3. Bucket files based on suffix (_train, _val, _test)
    data_buckets = {
        'train': {'sensor': [], 'ft': []},
        'val':   {'sensor': [], 'ft': []},
        'test':  {'sensor': [], 'ft': []}
    }

    for file_path in file_list:
        filename = os.path.basename(file_path)
        
        # Determine split type from filename
        if '_train.h5' in filename:
            split_type = 'train'
        elif '_val.h5' in filename:
            split_type = 'val'
        elif '_test.h5' in filename:
            split_type = 'test'
        else:
            print(f"Skipping file (unknown suffix): {filename}")
            continue

        # Load Data
        with h5py.File(file_path, 'r') as f:
            sensor_val = f['sensor_cal_data'][:]
            ft_val = f['ati_cal_FT'][:]
            
            data_buckets[split_type]['sensor'].append(sensor_val)
            data_buckets[split_type]['ft'].append(ft_val)
            print(f"Loaded {split_type}: {filename}")

    # 4. Stack Arrays for each split
    # Helper to stack or return empty if no data found
    def stack_data(key):
        if not data_buckets[key]['sensor']:
            print(f"Warning: No data found for split '{key}'")
            return None, None
        return (np.vstack(data_buckets[key]['sensor']), 
                np.vstack(data_buckets[key]['ft']))

    X_train, Y_train = stack_data('train')
    X_val, Y_val     = stack_data('val')
    X_test, Y_test   = stack_data('test')

    if X_train is None:
        print("Error: No training data found. Cannot compute normalization.")
        return

    print(f"\nData Split Dimensions:")
    
    if X_train is not None:
        print(f"Train | X: {X_train.shape}  Y: {Y_train.shape}")
    else:
        print("Train | Not found")

    if X_val is not None:
        print(f"Val   | X: {X_val.shape}  Y: {Y_val.shape}")
    else:
        print("Val   | Not found")

    if X_test is not None:
        print(f"Test  | X: {X_test.shape}  Y: {Y_test.shape}")
    else:
        print("Test  | Not found")

    # 5. Normalization (Z-Score) for INPUTS (X) AND TARGETS (Y)
    print("\nComputing normalization stats from Training set...")
    
    # --- INPUTS (Sensor Data) ---
    mean_X = np.mean(X_train, axis=0)
    std_X  = np.std(X_train, axis=0)
    std_X[std_X == 0] = 1.0  # Avoid division by zero

    # --- LABELS (Force/Torque Data) ---
    mean_Y = np.mean(Y_train, axis=0)
    std_Y  = np.std(Y_train, axis=0)
    std_Y[std_Y == 0] = 1.0  # Avoid division by zero

    # --- SAVE JSON CONSTANTS ---
    # We convert numpy arrays to python lists using .tolist() for JSON compatibility
    norm_constants = {
        "mu_x": mean_X.tolist(),
        "sd_x": std_X.tolist(),
        "mu_y": mean_Y.tolist(),
        "sd_y": std_Y.tolist()
    }

    json_path = os.path.join(SAVE_DIR, JSON_FILENAME)

    with open(json_path, 'w') as json_file:
        json.dump(norm_constants, json_file, indent=2)


    print(f"Saved normalization constants to: {json_path}")

    print("Normalizing X and Y data...")

    # Apply to Train
    X_train = (X_train - mean_X) / std_X
    Y_train = (Y_train - mean_Y) / std_Y

    # Apply to Val
    X_val = (X_val - mean_X) / std_X
    Y_val = (Y_val - mean_Y) / std_Y

    # Apply to Test
    X_test = (X_test - mean_X) / std_X
    Y_test = (Y_test - mean_Y) / std_Y

    # 6. Save Processed Datasets (HDF5)
    def save_h5(filename, x_data, y_data):
        if x_data is None: return
        
        full_path = os.path.join(SAVE_DIR, filename)
        with h5py.File(full_path, 'w') as f:
            f.create_dataset('data', data=x_data, compression='gzip')
            f.create_dataset('label', data=y_data, compression='gzip')
            # (Optional) We still save attrs in H5 for convenience, 
            # even though we have the JSON now.
            f.create_dataset('mean_X', data=mean_X)
            f.create_dataset('std_X', data=std_X)
            f.create_dataset('mean_Y', data=mean_Y)
            f.create_dataset('std_Y', data=std_Y)
            
        print(f"Saved: {full_path}")

    save_h5('train.h5', X_train, Y_train)
    save_h5('val.h5', X_val, Y_val)
    save_h5('test.h5', X_test, Y_test)

    print("\nProcessing complete.")

if __name__ == "__main__":
    process_data()