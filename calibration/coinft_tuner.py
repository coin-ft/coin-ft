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
        self.entry_tuning_file = ttk.Entry(control_frame, width=50) 
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

        # --- CHANGED: IDAC -> Ref ---
        ttk.Label(tune_frame, text="Ref:").grid(row=0, column=4, padx=5)
        self.entry_ref = ttk.Entry(tune_frame, width=10)
        self.entry_ref.grid(row=0, column=5, padx=5)

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

            # 2. Send Period
            samp_freq = 6
            samp_period = 1.0/samp_freq
            clock_freq = 100e3
            val = int(samp_period * clock_freq)
            p_bytes = [val % 256, val // 256] 
            
            self.ser.write(b'p')
            self.ser.write(bytes(p_bytes))
            time.sleep(0.1)

            # 3. Query Size
            self.ser.write(b'q')
            time.sleep(0.1)
            size_byte = self.ser.read(1)
            if not size_byte:
                raise Exception("No response from board (Step Q).")
            
            self.packet_size = ord(size_byte) - 1
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
                self.tuning_params = loaded_params.astype(int)
            except FileNotFoundError:
                messagebox.showerror("Error", "Tuning file not found.")
                self.ser.write(b'i')
                self.ser.close()
                return

            # 5. Send Tuning Commands
            for i in range(self.num_sensors):
                time.sleep(0.05)
                # Command: 't' + index + res + ref + comp
                # Note: self.tuning_params column 1 is Ref (formerly IDAC)
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
            
            ch_list = [f"Sns_{i}" for i in range(self.num_sensors)]
            self.channel_combo['values'] = ch_list
            self.channel_combo.current(0)
            self.on_channel_selected(None)
            self.parse_plot_list()

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
                self.ser.write(b'i') 
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
        while self.is_running and self.ser.is_open:
            try:
                b = self.ser.read(1)
                if b == b'\x02':
                    payload = self.ser.read(self.packet_size) 
                    
                    if len(payload) == self.packet_size and payload[-1] == 3:
                        sensor_bytes = payload[:-1] 
                        values = struct.unpack(f'<{self.num_sensors}H', sensor_bytes)
                        
                        if all(v < 60000 for v in values):
                            self.plot_data.append(values)
            except Exception as e:
                print(f"Read Error: {e}")
                break

    def animate_plot(self):
        """Updates the plot with dynamic Y-axis scaling."""
        if not self.is_running:
            return

        if len(self.plot_data) > 0:
            data_arr = np.array(self.plot_data)
            self.ax.clear()
            
            # Plot only active channels
            if self.active_channels:
                active_data = data_arr[:, self.active_channels]
                lines = self.ax.plot(active_data)
                self.ax.legend(lines, [f"Sn {i}" for i in self.active_channels], loc='upper left')
                
                # --- Dynamic Y-Axis Logic ---
                if active_data.size > 0:
                    current_min = np.min(active_data)
                    current_max = np.max(active_data)
                    
                    y_min = current_min - 50
                    y_max = current_max + 50
                    
                    # Prevent crash if flat signal
                    if y_min == y_max:
                        y_min -= 10
                        y_max += 10
                        
                    self.ax.set_ylim(y_min, y_max)
                # -----------------------------
            
            self.ax.grid(True)
            self.canvas.draw()

        self.root.after(100, self.animate_plot)

    def on_channel_selected(self, event):
        idx = self.channel_combo.current()
        if idx >= 0 and self.tuning_params is not None:
            params = self.tuning_params[idx]
            # Column 0: Res, Column 1: Ref (IDAC), Column 2: Comp
            self.entry_res.delete(0, tk.END); self.entry_res.insert(0, str(params[0]))
            
            # --- CHANGED: IDAC -> Ref ---
            self.entry_ref.delete(0, tk.END); self.entry_ref.insert(0, str(params[1]))
            
            self.entry_comp.delete(0, tk.END); self.entry_comp.insert(0, str(params[2]))

    def update_params(self):
        """Sends new parameters to the firmware."""
        if not self.is_running: return
        
        try:
            idx = self.channel_combo.current()
            res = int(self.entry_res.get())
            
            # --- CHANGED: IDAC -> Ref ---
            ref = int(self.entry_ref.get())
            
            comp = int(self.entry_comp.get())
            
            # Update local memory
            self.tuning_params[idx] = [res, ref, comp]
            
            # Send Command
            # 't', sensorIdx, res, ref, comp
            cmd = struct.pack('BBbbb', ord('t'), idx, res, ref, comp)
            self.ser.write(cmd)
            self.ser.flush()
            print(f"Updated Ch {idx}: {res}, {ref}, {comp}")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric input.")

    def save_params(self):
        if self.tuning_params is None: return
        
        mat_path = os.path.join(self.config_dir, 'Tuning_saved.mat')
        txt_path = os.path.join(self.config_dir, 'Tuning_saved.txt')

        scipy.io.savemat(mat_path, {'TuningParams': self.tuning_params})
        
        with open(txt_path, 'w') as f:
            f.write('const uint8 Params_All[{}][3] = {{\n'.format(self.num_sensors))
            for i in range(self.num_sensors):
                row = self.tuning_params[i]
                f.write(f'\t{{{row[0]},{row[1]},{row[2]}}},\n')
            f.write('};\n')
            
        messagebox.showinfo("Success", "Parameters saved to Tuning_saved.mat and .txt")

    def parse_plot_list(self):
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

if __name__ == "__main__":
    root = tk.Tk()
    app = CoinFTTuner(root)
    
    def on_closing():
        app.stop_system()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()