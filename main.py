import serial
import time
from datetime import datetime
import csv
import os
import queue
import threading
from GUI import start_gui
class PulseSensorCollector:
    def __init__(self, port='COM3', baudrate=115200,save_dir="./data/rawsignal"):
        """
        初始化串口采集器 / Initialize serial collector

        参数 / Parameters:
            port: 串口号 (Windows: COM3, COM4... | Linux/Mac: /dev/ttyUSB0, /dev/ttyACM0...)
            baudrate: 波特率，默认115200
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

    def connect(self):
        """连接串口 / Connect to serial port"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # 等待Arduino重启 / Wait for Arduino reset
            print(f"✅ 已连接到 {self.port} (波特率: {self.baudrate})")
            print(f"✅ Connected to {self.port} (Baudrate: {self.baudrate})")
            return True
        except serial.SerialException as e:
            print(f"❌ 串口连接失败 / Serial connection failed: {e}")
            print(f"💡 提示 / Tip: 请检查串口号是否正确 / Please check if port name is correct")
            return False

    def send_command(self, command):
        """发送命令到Arduino / Send command to Arduino"""
        if self.ser and self.ser.is_open:
            self.ser.write(f"{command}\n".encode())
            print(f"Sent command: {command}")
        else:
            print("Serial not connected")

    def start_collection(self,save_dir=None):
        """
        开始采集数据。将在./data/rawsignal/下创建一个时间戳文件夹，并在其中保存pulse_data.csv
        Start data collection. Will create a timestamped folder under ./data/rawsignal/
        and save pulse_data.csv inside it.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        target_dir = os.path.join("./data/rawsignal", timestamp)

        os.makedirs(target_dir, exist_ok=True)

        filename = os.path.join(target_dir, "pulse_data.csv")
        self.csv_file = open(filename, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)

        # 写入您指定的表头 / Write the header you specified
        self.csv_writer.writerow([
            'PC_Timestamp_ms',  # You know
            'PC_DateTime',  # You know
            'Arduino_millis',  # Arduino run time
            'Signal_Value',  # PPG Value
            'Package_Num'  # You know
        ])

        self.collection_active = True
        print(f"Started data collection, saving to: {filename}")
        print("-" * 60)

    def stop_collection(self):
        if self.csv_file:
            self.csv_file.close()
            self.collection_active = False
            print("-" * 60)
            print(f"Finished data collection, saving to: {self.csv_file.name}")
            print(f"✅ Collection completed, file saved")

    def parse_collect_line(self, line):
        """解析采集数据行 / Parse collection data line"""
        try:
            # 格式: [COLLECT] TIMESTAMP_REQUEST | 5234 | 512 | 128
            if "TIMESTAMP_REQUEST" in line:
                parts = line.split("|")
                if len(parts) >= 4:
                    arduino_millis = parts[1].strip()
                    signal_value = parts[2].strip()
                    led_output = parts[3].strip()

                    # 获取电脑时间戳（毫秒精度）/ Get PC timestamp (millisecond precision)
                    pc_timestamp_ms = int(time.time() * 1000)
                    pc_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    return [pc_timestamp_ms, pc_datetime, arduino_millis, signal_value, led_output]
        except Exception as e:
            print(f"⚠️ 解析错误 / Parse error: {e}")
        return None

    def input_thread(self):
        """独立线程处理用户输入 / Separate thread for user input"""
        while self.running:
            try:
                user_input = input().strip()
                if user_input:
                    self.command_queue.put(user_input)
            except EOFError:
                break
            except Exception as e:
                print(f"⚠️ 输入错误 / Input error: {e}")

    def process_command(self, user_input):
        """处理用户命令 / Process user command"""
        if user_input.lower() == 'quit':
            self.running = False
            print("\n👋 正在退出... / Exiting...")
            return False

        # 如果输入collect，准备开始采集 / If collect, prepare to start collection
        if user_input.lower() == 'collect':
            self.send_command('collect')
            time.sleep(0.5)  # 等待Arduino响应 / Wait for Arduino response
            self.start_collection(save_dir=self.save_dir)
        else:
            self.send_command(user_input)

        return True

    def run(self):
        """主运行循环 / Main run loop"""
        print("\n" + "=" * 60)
        print("🎯 脉搏传感器数据采集系统 / Pulse Sensor Data Collection System")
        print("=" * 60)
        print("\n可用命令 / Available commands:")
        print("  pause   Pause monitoring")
        print("  start   Start monitoring")
        print("  collect - Collect data for 10 seconds")
        print("  0-255   - Set LED brightness")
        print("  quit    - Exit program")
        print("\n请输入命令 / Enter command:")
        print("-" * 60)
        input_thread = threading.Thread(target=self.input_thread, daemon=True)
        input_thread.start()
        try:
            while self.running:
                # 处理命令队列 / Process command queue
                try:
                    user_input = self.command_queue.get_nowait()
                    if not self.process_command(user_input):
                        break
                except queue.Empty:
                    pass

                # 读取串口数据 / Read serial data
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()

                    if line:
                        # 显示原始数据 / Display raw data
                        print(line)

                        # 如果是采集数据，解析并保存 / If collection data, parse and save
                        if self.collection_active and "[COLLECT]" in line:
                            data = self.parse_collect_line(line)
                            if data:
                                self.csv_writer.writerow(data)
                                self.csv_file.flush()  # 立即写入文件 / Write to file immediately

                        # 检测采集完成 / Detect collection completion
                        if "COLLECTION COMPLETED" in line:
                            self.stop_collection()

                time.sleep(0.01)  # 短暂延迟，减少CPU占用 / Short delay to reduce CPU usage

        except KeyboardInterrupt:
            print("\n\n User interrupted (Ctrl+C)")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源 / Cleanup resources"""
        if self.collection_active:
            self.stop_collection()
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("🔌 串口已关闭 / Serial port closed")


def main():
    """主函数 / Main function"""
    print("=" * 60)
    print("Configure Serial Port")
    print("=" * 60)
    port = 'COM3'

    # 创建采集器实例 / Create collector instance
    collector = PulseSensorCollector(port=port, baudrate=115200)

    # 连接串口 / Connect to serial
    if collector.connect():
        # 调用一行代码启动GUI，并将collector实例传递给它
        # Call one line of code to start the GUI and pass the collector instance to it
        start_gui(collector)
    else:
        # 在GUI启动前处理连接失败的情况
        # Handle connection failure before starting the GUI
        print("\n 无法启动，请检查串口设置 / Cannot start, please check serial settings")
        # 可以在这里显示一个错误弹窗
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        tk.messagebox.showerror("Connection Error", f"Failed to connect to {port}. Please check serial settings.")


if __name__ == "__main__":
    main()