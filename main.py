import serial
import time
from datetime import datetime
import csv
import os
import queue
import threading
import re
from GUI import start_gui
from realtimemonitor import start_realtime_monitor


class PulseSensorCollector:
    def __init__(self, port='COM3', baudrate=115200, save_dir="./data/rawsignal"):
        """
        Initialize serial collector.

        Parameters:
            port: Serial port name (Windows: COM3... | Linux/Mac: /dev/ttyUSB0, /dev/ttyACM0...)
            baudrate: Baud rate (default 115200)
        """
        self.port = port
        self.baudrate = baudrate
        self.save_dir = save_dir
        self.ser = None
        self.csv_file = None
        self.csv_writer = None
        self.collection_active = False
        self.running = True
        self.command_queue = queue.Queue()
        self.monitor = None  # Real-time monitor window reference
        self.is_paused = False  # Track Arduino pause status

    def connect(self):
        """Connect to serial port"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            print(f"Connected to {self.port} (Baudrate: {self.baudrate})")
            return True
        except serial.SerialException as e:
            print(f"Serial connection failed: {e}")
            print(f"Please check if port name is correct")
            return False

    def send_command(self, command):
        """Send command to Arduino"""
        if self.ser and self.ser.is_open:
            self.ser.write(f"{command}\n".encode())
            print(f"Sent command: {command}")
        else:
            print("Serial not connected")

    def start_collection(self, save_dir=None):
        """
        Start data collection.
        Creates a timestamped folder under ./data/rawsignal/
        and saves pulse_data.csv inside it.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = os.path.join("./data/rawsignal", timestamp)
        os.makedirs(target_dir, exist_ok=True)

        filename = os.path.join(target_dir, "pulse_data.csv")
        self.csv_file = open(filename, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            'PC_Timestamp_ms',
            'PC_DateTime',
            'Arduino_millis',
            'Signal_Value',
            'Package_Num'
        ])

        self.collection_active = True
        print(f"Started data collection, saving to: {filename}")
        print("-" * 60)

    def stop_collection(self):
        if self.csv_file:
            self.csv_file.close()
            self.collection_active = False
            print("-" * 60)
            print(f"Finished data collection, saved to: {self.csv_file.name}")
            print(f"Collection completed")

    def parse_collect_line(self, line):
        """Parse data line during collection"""
        try:
            # Format: [COLLECT] TIMESTAMP_REQUEST | 5234 | 512 | 128
            if "TIMESTAMP_REQUEST" in line:
                parts = line.split("|")
                if len(parts) >= 4:
                    arduino_millis = parts[1].strip()
                    signal_value = parts[2].strip()
                    led_output = parts[3].strip()

                    pc_timestamp_ms = int(time.time() * 1000)
                    pc_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    return [pc_timestamp_ms, pc_datetime, arduino_millis, signal_value, led_output]
        except Exception as e:
            print(f"Parse error: {e}")
        return None

    def parse_signal_from_line(self, line):
        """
        Extract signal value from serial data line for real-time monitoring.

        Supported formats:
        - [COLLECT] TIMESTAMP_REQUEST | 5234 | 512 | 128
        - [SENSOR] Signal: 512 | LED Output: 128 | Package Number: 0%
        """
        try:
            if "[COLLECT]" in line and "TIMESTAMP_REQUEST" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    signal = int(parts[2].strip())
                    return signal

            elif "[SENSOR]" in line:
                match = re.search(r'Signal:\s*(\d+)', line)
                if match:
                    return int(match.group(1))

        except Exception:
            pass

        return None

    def input_thread(self):
        """Thread for user input"""
        while self.running:
            try:
                user_input = input().strip()
                if user_input:
                    self.command_queue.put(user_input)
            except EOFError:
                break
            except Exception as e:
                print(f"Input error: {e}")

    def process_command(self, user_input):
        """Process user command"""
        if user_input.lower() == 'quit':
            self.running = False
            print("\nExiting...")
            return False

        if user_input.lower() == 'collect':
            self.send_command('collect')
            time.sleep(0.5)
            self.start_collection(save_dir=self.save_dir)
        else:
            self.send_command(user_input)

        return True

    def run(self):
        """Main run loop"""
        print("\n" + "=" * 60)
        print("Pulse Sensor Data Collection System")
        print("=" * 60)
        print("\nAvailable commands:")
        print("  pause   - Pause monitoring")
        print("  start   - Start monitoring")
        print("  collect - Collect data for 10 seconds")
        print("  0-255   - Set LED brightness")
        print("  quit    - Exit program")
        print("\nEnter command:")
        print("-" * 60)

        input_thread = threading.Thread(target=self.input_thread, daemon=True)
        input_thread.start()

        try:
            while self.running:
                # Process queued user commands
                try:
                    user_input = self.command_queue.get_nowait()
                    if not self.process_command(user_input):
                        break
                except queue.Empty:
                    pass

                # Read serial data
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()

                    if line:
                        print(line)

                        if "[SYSTEM]" in line:
                            if "PAUSED" in line:
                                self.is_paused = True
                                self.ser.reset_input_buffer()
                                print("Buffer cleared")
                            elif "STARTED" in line:
                                self.is_paused = False

                        if self.monitor and not self.is_paused:
                            signal_value = self.parse_signal_from_line(line)
                            if signal_value is not None:
                                self.monitor.add_data_point(signal_value)

                        if self.collection_active and "[COLLECT]" in line:
                            data = self.parse_collect_line(line)
                            if data:
                                self.csv_writer.writerow(data)
                                self.csv_file.flush()

                        if "COLLECTION COMPLETED" in line:
                            self.stop_collection()

                else:
                    time.sleep(0.0001)

        except KeyboardInterrupt:
            print("\nUser interrupted (Ctrl+C)")

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        if self.collection_active:
            self.stop_collection()
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed")


def main():
    """Main function"""
    print("=" * 60)
    print("Configure Serial Port")
    print("=" * 60)

    port = 'COM3'
    collector = PulseSensorCollector(port=port, baudrate=115200)

    if collector.connect():
        monitor = start_realtime_monitor(collector)
        collector.monitor = monitor

        start_gui(collector)
    else:
        print("\nCannot start, please check serial settings")
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        tk.messagebox.showerror("Connection Error",
                                f"Failed to connect to {port}. Please check serial settings.")


if __name__ == "__main__":
    main()
