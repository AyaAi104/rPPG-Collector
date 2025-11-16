import tkinter as tk
from tkinter import ttk
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import threading
from collections import deque
import re


class RealtimePPGMonitor:
    def __init__(self, collector, max_points=500):
        """
        Real-time PPG waveform monitor.

        Parameters:
            collector: PulseSensorCollector instance
            max_points: Maximum number of displayed data points (default 500, ~10 seconds)
        """
        self.collector = collector
        self.max_points = max_points

        # Data buffers
        self.signal_data = deque(maxlen=max_points)
        self.time_data = deque(maxlen=max_points)

        # Relative time counter (seconds)
        self.time_counter = 0
        self.data_interval = 0.02  # 20 ms sampling interval

        # Statistics
        self.current_signal = 0
        self.max_signal = 0
        self.min_signal = 1023
        self.avg_signal = 0
        self.data_count = 0

        # Window state
        self.running = True
        self.root = None

    def create_window(self):
        """Create monitor window as an independent Toplevel window."""
        # Create an invisible root window if needed
        if not hasattr(tk, '_default_root') or tk._default_root is None:
            hidden_root = tk.Tk()
            hidden_root.withdraw()

        self.root = tk.Toplevel()
        self.root.title("Real-time PPG Signal Monitor")
        self.root.geometry("1000x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Grid weight
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # ===== Top info panel =====
        info_frame = ttk.LabelFrame(main_frame, text="Signal Statistics", padding="10")
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Statistics labels
        self.current_label = ttk.Label(info_frame, text="Current: --", font=("Arial", 12))
        self.current_label.grid(row=0, column=0, padx=20)

        self.max_label = ttk.Label(info_frame, text="Max: --", font=("Arial", 12))
        self.max_label.grid(row=0, column=1, padx=20)

        self.min_label = ttk.Label(info_frame, text="Min: --", font=("Arial", 12))
        self.min_label.grid(row=0, column=2, padx=20)

        self.avg_label = ttk.Label(info_frame, text="Average: --", font=("Arial", 12))
        self.avg_label.grid(row=0, column=3, padx=20)

        self.count_label = ttk.Label(info_frame, text="Points: 0", font=("Arial", 12))
        self.count_label.grid(row=0, column=4, padx=20)

        # ===== Plot area =====
        plot_frame = ttk.LabelFrame(main_frame, text="PPG Waveform", padding="5")
        plot_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 5), dpi=100)
        self.fig.patch.set_facecolor('#f0f0f0')

        # Empty line
        self.line, = self.ax.plot([], [], 'r-', linewidth=2, label='PPG Signal')

        # Axes settings
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 1023)
        self.ax.set_xlabel('Time (seconds)', fontsize=12)
        self.ax.set_ylabel('Signal Value (0-1023)', fontsize=12)
        self.ax.set_title('Real-time PPG Waveform', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.legend(loc='upper right')

        # Embed into Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ===== Bottom status bar =====
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.status_label = ttk.Label(
            status_frame,
            text="Monitoring...",
            font=("Arial", 10),
            foreground="green"
        )
        self.status_label.pack(side=tk.LEFT)

        # Start animation (no mainloop here; managed by outer GUI)
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=False)

    def add_data_point(self, signal_value):
        """Add a data point from external caller (e.g., main program)."""
        if signal_value is not None:
            # Add to buffers
            self.signal_data.append(signal_value)
            self.time_data.append(self.time_counter)
            self.time_counter += self.data_interval

            # Update statistics
            self.current_signal = signal_value
            self.max_signal = max(self.max_signal, signal_value)
            self.min_signal = min(self.min_signal, signal_value)
            self.data_count += 1

            # Moving average
            if len(self.signal_data) > 0:
                self.avg_signal = sum(self.signal_data) / len(self.signal_data)

    def update_plot(self, frame):
        """Callback for updating the plot."""
        if len(self.signal_data) > 0:
            # Update curve data
            self.line.set_data(list(self.time_data), list(self.signal_data))

            # Fixed 10-second window
            window_size = 10.0
            if len(self.time_data) > 0:
                current_time = self.time_data[-1]
                self.ax.set_xlim(current_time - window_size, current_time)

            # Update labels
            if self.root:
                try:
                    self.current_label.config(text=f"Current: {self.current_signal}")
                    self.max_label.config(text=f"Max: {self.max_signal}")
                    self.min_label.config(text=f"Min: {self.min_signal}")
                    self.avg_label.config(text=f"Average: {int(self.avg_signal)}")
                    self.count_label.config(text=f"Points: {self.data_count}")
                except:
                    pass

        return self.line,

    def on_closing(self):
        """Window close callback."""
        self.running = False
        if self.root:
            self.root.destroy()
        print("Realtime monitor window closed")


def start_realtime_monitor(collector):
    """
    Start real-time PPG monitor window.

    Parameters:
        collector: PulseSensorCollector instance

    Returns:
        RealtimePPGMonitor instance
    """
    monitor = RealtimePPGMonitor(collector)
    monitor.create_window()
    return monitor
