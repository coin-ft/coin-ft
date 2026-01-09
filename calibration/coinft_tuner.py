import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import serial
import serial.tools.list_ports
import numpy as np
import scipy.io
import struct
import threading
import time
from collections import deque
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os

class CoinFTTuner:
    def __init__(self, root):
        self.root = root
        self.root.title("CoinFT Tuner (Python)")
        self.root.geometry("1000x700")

        # --- System State Variables ---
        self.ser = None
        self.is_running = False
        self.read_thread = None
        self.tuning_params = None # Will be (num_sensors x 3) numpy array
        self.num_sensors = 0
        self.packet_size = 0
        self.plot_data = deque(maxlen=500) # Circular buffer for plotting
        self.active_channels = [] # List of channels to plot
        
        # --- UI Layout ---
        self.create_layout()
        
        # Scan ports on startup
        self.refresh_ports()

    def create_layout(self):
        # Top Control Panel
        control_frame = ttk.LabelFrame(self.root, text="Connection & Setup")
        control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(control_frame, text="Port:").pack(side="left", padx=5)
        self.port_combo = ttk.Combobox(control_frame, width=15)
        self.port_combo.pack(side="left", padx=5)
        
        self.btn_refresh = ttk.Button(control_frame, text="Refresh", command=self.refresh_ports)
        self.btn_refresh.pack(side="left", padx=5)

        ttk.Label(control_frame, text="Tuning File (.mat):").pack(side="left", padx=5)
        self.entry_tuning_file = ttk.Entry(control_frame, width=50) # Increased width for path
        self.entry_tuning_file.pack(side="left", padx=5)
        
        # Calculate path: active_script_dir/../hardware_configs
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_dir = os.path.abspath(os.path.join(base_dir, "..", "hardware_configs"))
        default_path = os.path.join(self.config_dir, "Tuning_saved.mat")
        
        self.entry_tuning_file.insert(0, default_path)
        
        self.btn_start = ttk.Button(control_frame, text="Start", command=self.start_system)
        self.btn_start.pack(side="left", padx=5)
        
        self.btn_stop = ttk.Button(control_frame, text="Stop", command=self.stop_system, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        # Plot Area
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Sensor Data Stream")
        self.ax.set_xlabel("Time (samples)")
        self.ax.set_ylabel("Raw Value")
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Bottom Tuning Panel
        tune_frame = ttk.LabelFrame(self.root, text="Parameter Tuning")
        tune_frame.pack(fill="x", padx=5, pady=5)

        # Channel Selector
        ttk.Label(tune_frame, text="Select Channel:").grid(row=0, column=0, padx=5, pady=5)
        self.channel_combo = ttk.Combobox(tune_frame, state="readonly")
        self.channel_combo.grid(row=0, column=1, padx=5, pady=5)
        self.channel_combo.bind("<<ComboboxSelected>>", self.on_channel_selected)

        # Parameters
        ttk.Label(tune_frame, text="Resolution:").grid(row=0, column=2, padx=5)
        self.entry_res = ttk.Entry(tune_frame, width=10)
        self.entry_res.grid(row=0, column=3, padx=5)

        ttk.Label(tune_frame, text="IDAC:").grid(row=0, column=4, padx=5)
        self.entry_idac = ttk.Entry(tune_frame, width=10)
        self.entry_idac.grid(row=0, column=5, padx=5)

        ttk.Label(tune_frame, text="Compensation:").grid(row=0, column=6, padx=5)
        self.entry_comp = ttk.Entry(tune_frame, width=10)
        self.entry_comp.grid(row=0, column=7, padx=5)

        # Buttons
        self.btn_update = ttk.Button(tune_frame, text="Update Params", command=self.update_params, state="disabled")
        self.btn_update.grid(row=0, column=8, padx=10)

        self.btn_save = ttk.Button(tune_frame, text="Save to File", command=self.save_params, state="disabled")
        self.btn_save.grid(row=0, column=9, padx=10)
        
        # Plot Selection
        plot_ctrl_frame = ttk.Frame(self.root)
        plot_ctrl_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(plot_ctrl_frame, text="Plot Channels (e.g. '0-5', 'all', '1,3'):").pack(side="left")
        self.entry_plot_list = ttk.Entry(plot_ctrl_frame)
        self.entry_plot_list.pack(side="left", fill="x", expand=True, padx=5)
        self.entry_plot_list.insert(0, "all")
        self.btn_plot_upd = ttk.Button(plot_ctrl_frame, text="Update Plot List", command=self.parse_plot_list)
        self.btn_plot_upd.pack(side="left")

    def refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)

    def start_system(self):
        port = self.port_combo.get()
        tuning_file = self.entry_tuning_file.get()
        
        if not port:
            messagebox.showerror("Error", "No COM port selected.")
            return

        try:
            # 1. Open Serial
            self.ser = serial.Serial(port, 1000000, timeout=1)
            self.ser.reset_input_buffer()
            time.sleep(0.1)
            print(f"Opened {port}")

            # 2. Send Period (Handshake Step 1)
            # Replicating MATLAB: 6Hz -> ~0.166s
            # KitprogTimerClockFreq = 100e3
            # Period calculation from MATLAB logic
            samp_freq = 6
            samp_period = 1.0/samp_freq
            clock_freq = 100e3
            val = int(samp_period * clock_freq)
            # PeriodInput bytes: [Low Byte, High Byte]
            p_bytes = [val % 256, val // 256] 
            
            self.ser.write(b'p')
            self.ser.write(bytes(p_bytes))
            time.sleep(0.1)

            # 3. Query Size (Handshake Step 2)
            self.ser.write(b'q')
            time.sleep(0.1)
            size_byte = self.ser.read(1)
            if not size_byte:
                raise Exception("No response from board (Step Q).")
            
            self.packet_size = ord(size_byte) - 1 # MATLAB logic: fread(1)-1
            
            # MATLAB: handles.num_sensors = (handles.packet_size - 1) / 2 (Assuming temp_comp=False)
            self.num_sensors = int((self.packet_size - 1) / 2)
            print(f"Packet Size: {self.packet_size}, Sensors: {self.num_sensors}")

            # 4. Load Tuning Params
            try:
                mat = scipy.io.loadmat(tuning_file)
                loaded_params = mat['TuningParams']
                
                if loaded_params.shape[0] != self.num_sensors:
                    messagebox.showerror("Error", f"Tuning file mismatch. Expected {self.num_sensors} rows.")
                    self.ser.write(b'i')
                    self.ser.close()
                    return
                self.tuning_params = loaded_params.astype(int) # Ensure integers
            except FileNotFoundError:
                messagebox.showerror("Error", "Tuning file not found.")
                self.ser.write(b'i')
                self.ser.close()
                return

            # 5. Send Tuning Commands
            for i in range(self.num_sensors):
                time.sleep(0.05)
                # Command: 't' + index + res + idac + comp
                cmd = struct.pack('BBbbb', 
                                  ord('t'), 
                                  i, 
                                  self.tuning_params[i,0], 
                                  self.tuning_params[i,1], 
                                  self.tuning_params[i,2])
                self.ser.write(cmd)
            
            print("Tuning Params Sent.")
            time.sleep(0.5)

            # 6. Start Streaming
            self.ser.write(b's')
            time.sleep(0.1)

            # 7. UI Updates
            self.is_running = True
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
            self.btn_update.config(state="normal")
            self.btn_save.config(state="normal")
            
            # Populate Channel Combo
            ch_list = [f"Sns_{i}" for i in range(self.num_sensors)]
            self.channel_combo['values'] = ch_list
            self.channel_combo.current(0)
            self.on_channel_selected(None)
            self.parse_plot_list() # Init active channels

            # 8. Start Thread
            self.read_thread = threading.Thread(target=self.serial_reader_thread)
            self.read_thread.daemon = True
            self.read_thread.start()

            # 9. Start Plot Animation Loop
            self.animate_plot()

        except Exception as e:
            messagebox.showerror("Connection Error", str(e))
            if self.ser and self.ser.is_open:
                self.ser.close()

    def stop_system(self):
        self.is_running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b'i') # Idle command
                time.sleep(0.2)
                self.ser.close()
            except:
                pass
        
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_update.config(state="disabled")
        self.btn_save.config(state="disabled")
        print("System Stopped.")

    def serial_reader_thread(self):
        """Background thread to read data continuously."""
        while self.is_running and self.ser.is_open:
            try:
                # MATLAB logic: wait for 2 (Start byte), then read packet
                # Efficient Python way: read until we find 0x02
                b = self.ser.read(1)
                if b == b'\x02':
                    # Read the rest of the packet
                    # Packet size includes start/end bytes in MATLAB logic, but we already read start.
                    # MATLAB: data = fread(com, packet_size)
                    # We need to read (packet_size) bytes roughly.
                    # Based on logic: packet_size was derived from 'q'.
                    # Let's assume packet_size is the FULL payload length.
                    
                    payload = self.ser.read(self.packet_size) 
                    
                    if len(payload) == self.packet_size and payload[-1] == 3: # End byte 0x03
                        # Parse Data
                        # MATLAB: data(byte_num)+256*data(byte_num+1) -> Little Endian Short
                        # payload[0:-1] contains the sensor data bytes
                        sensor_bytes = payload[:-1] 
                        
                        # Unpack unsigned shorts (H) little endian (<)
                        # num_sensors * 2 bytes
                        values = struct.unpack(f'<{self.num_sensors}H', sensor_bytes)
                        
                        # Filter (False Threshold logic from MATLAB)
                        if all(v < 60000 for v in values):
                            self.plot_data.append(values)
            except Exception as e:
                print(f"Read Error: {e}")
                break

    def animate_plot(self):
        """Updates the plot on the main UI thread."""
        if not self.is_running:
            return

        if len(self.plot_data) > 0:
            data_arr = np.array(self.plot_data)
            self.ax.clear()
            
            # Plot only active channels
            if self.active_channels:
                # Slice the array to get only the data we are actually plotting
                active_data = data_arr[:, self.active_channels]
                
                lines = self.ax.plot(active_data)
                self.ax.legend(lines, [f"Sn {i}" for i in self.active_channels], loc='upper left')
                
                # --- Dynamic Y-Axis Logic ---
                if active_data.size > 0:
                    # Calculate min/max of ONLY the currently visible data
                    current_min = np.min(active_data)
                    current_max = np.max(active_data)
                    
                    # Apply the user-requested buffer
                    # You can add max(0, ...) here if you never want it to go negative
                    y_min = current_min - 50
                    y_max = current_max + 50
                    
                    # Safety check: if signal is perfectly flat, give it a small range
                    if y_min == y_max:
                        y_min -= 10
                        y_max += 10
                        
                    self.ax.set_ylim(y_min, y_max)
                # -----------------------------
            
            self.ax.grid(True)
            self.canvas.draw()

        # Schedule next update (100ms = 10FPS)
        self.root.after(100, self.animate_plot)

    def on_channel_selected(self, event):
        idx = self.channel_combo.current()
        if idx >= 0 and self.tuning_params is not None:
            params = self.tuning_params[idx]
            self.entry_res.delete(0, tk.END); self.entry_res.insert(0, str(params[0]))
            self.entry_idac.delete(0, tk.END); self.entry_idac.insert(0, str(params[1]))
            self.entry_comp.delete(0, tk.END); self.entry_comp.insert(0, str(params[2]))

    def update_params(self):
        """Sends new parameters to the firmware."""
        if not self.is_running: return
        
        try:
            idx = self.channel_combo.current()
            res = int(self.entry_res.get())
            idac = int(self.entry_idac.get())
            comp = int(self.entry_comp.get())
            
            # Update local memory
            self.tuning_params[idx] = [res, idac, comp]
            
            # Send Command
            # 't', sensorIdx, res, idac, comp
            cmd = struct.pack('BBbbb', ord('t'), idx, res, idac, comp)
            self.ser.write(cmd)
            self.ser.flush()
            print(f"Updated Ch {idx}: {res}, {idac}, {comp}")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric input.")

    def save_params(self):
        """Saves current parameters to MAT and TXT."""
        if self.tuning_params is None: return
        
        # Construct absolute paths
        mat_path = os.path.join(self.config_dir, 'Tuning_saved.mat')
        txt_path = os.path.join(self.config_dir, 'Tuning_saved.txt')

        # 1. Save MAT
        scipy.io.savemat(mat_path, {'TuningParams': self.tuning_params})
        
        # 2. Save TXT (C Struct style)
        with open(txt_path, 'w') as f:
            f.write('const uint8 Params_All[{}][3] = {{\n'.format(self.num_sensors))
            for i in range(self.num_sensors):
                row = self.tuning_params[i]
                f.write(f'\t{{{row[0]},{row[1]},{row[2]}}},\n')
            f.write('};\n')
            
        messagebox.showinfo("Success", "Parameters saved to Tuning_saved.mat and .txt")

    def parse_plot_list(self):
        """Parses the '0-5', 'all' string for plotting."""
        txt = self.entry_plot_list.get().lower().replace(" ", "")
        new_list = []
        
        if txt == 'all':
            new_list = list(range(self.num_sensors))
        else:
            parts = txt.split(',')
            for p in parts:
                if '-' in p:
                    try:
                        start, end = map(int, p.split('-'))
                        if start <= end < self.num_sensors:
                            new_list.extend(range(start, end + 1))
                    except: pass
                else:
                    try:
                        val = int(p)
                        if val < self.num_sensors:
                            new_list.append(val)
                    except: pass
        
        self.active_channels = sorted(list(set(new_list)))

# --- Main Entry Point ---
if __name__ == "__main__":
    root = tk.Tk()
    app = CoinFTTuner(root)
    
    # Handle clean exit
    def on_closing():
        app.stop_system()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()