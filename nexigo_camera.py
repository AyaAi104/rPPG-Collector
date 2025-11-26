import cv2
import time
import os
import threading
import numpy as np
from config import data_settings as settings
from queue import Queue
from utils.distance_ruler import FaceDistanceMeasurement, create_optimal_calibration, CameraCalibration
from pathlib import Path
from utils.pixel_counter import FacePixelCounter

window_name = "Camera Preview"


class Camera:
    def __init__(self, camera_index):
        self.frame_count = 0
        self.output_dir = None
        self.TARGET_FPS = 50.0
        self.FRAME_INTERVAL = 1.0 / self.TARGET_FPS  # 0.02s
        self.RECORD_DURATION = settings["record_duration"] + 0.3 # for warm up, otherwise, the first frame will be fully black
        #self.MAX_FRAMES = int(self.TARGET_FPS * self.RECORD_DURATION)
        self.MAX_FRAMES = int(self.TARGET_FPS * settings["record_duration"])+1
        # Save
        self.save_queue = Queue()
        self.save_thread = None
        self.save_thread_running = False

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.is_window_created = False
        if not self.cap.isOpened():
            print("Still cannot open camera, check if other apps are using it.")
            exit()
        self.cap.set(cv2.CAP_PROP_FPS, self.TARGET_FPS)
        print("Camera reported FPS:", self.cap.get(cv2.CAP_PROP_FPS))
        #
        calibration = self._load_calibration(settings["calibration_file"])
        self.measurer = FaceDistanceMeasurement(calibration)

        self.pixel_counter = FacePixelCounter()

    def _load_calibration(self, calibration_file = settings["calibration_file"]):
        """"""
        calibration_path = Path("./data/camera_parameter") / calibration_file

        if calibration_path.exists():
            try:
                print(f"Loading calibration from: {calibration_file}")
                data = np.load(calibration_path)

                # Extract camera matrix parameters
                camera_matrix = data['camera_matrix']
                fx = camera_matrix[0, 0]
                fy = camera_matrix[1, 1]
                cx = camera_matrix[0, 2]
                cy = camera_matrix[1, 2]

                # Use average focal length (fx and fy should be similar)
                focal_length = (fx + fy) / 2.0

                image_width = int(data['image_width'])
                image_height = int(data['image_height'])

                print(" Calibration loaded successfully at {}!".format(calibration_file))
                print(f"  Focal Length (fx): {fx:.2f} px")
                print(f"  Focal Length (fy): {fy:.2f} px")
                print(f"  Mean Focal Length: {focal_length:.2f} px")
                print(f"  Principal Point: ({cx:.2f}, {cy:.2f})")
                print(f"  Image Size: {image_width}x{image_height}")

                # Create CameraCalibration object
                calibration = CameraCalibration(
                    focal_length=focal_length,
                    principal_point=(cx, cy),
                    image_width=image_width,
                    image_height=image_height
                )

                return calibration

            except Exception as e:
                print(f" Failed to load calibration file: {e}")
                print("  Falling back to default calibration...")
        else:
            print(f" Calibration file not found: {calibration_file}")
            print("  Falling back to default calibration...")

                # Fallback to create_optimal_calibration
        print("Using create_optimal_calibration as fallback...")
        calibration = create_optimal_calibration(self.cap, 720)
        return calibration

    def _save_worker(self):
        """"""
        while self.save_thread_running:
            try:
                item = self.save_queue.get(timeout=0.1)
                if item is None:
                    break
                filename, frame = item
                cv2.imwrite(filename, frame)
                self.save_queue.task_done()
            except:
                continue

    def preview(self):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self.is_window_created = True
        while True:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user (X).")
                break

            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Exit by key")
                break
    def warmup(self):
        print("Warming up camera...")
        for _ in range(5):
            ret, frame = self.cap.read()

        print("Camera ready!")

    def record(self, record_time=10):

        self.save_thread_running = True
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        if not self.is_window_created:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            self.is_window_created = True

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        base_dir = "./data/video"
        os.makedirs(base_dir, exist_ok=True)

        session_ms = int(time.time() * 1000)
        self.output_dir = os.path.join(base_dir, f"frames_{session_ms}")
        os.makedirs(self.output_dir, exist_ok=True)

        print("Saving frames to folder:", self.output_dir)

        start_time = time.time()
        next_capture_time = start_time
        self.frame_count = 0

        print(f"Start recording: {record_time}s, target ~{int(self.TARGET_FPS * record_time)} frames")
        round = 0
        while True:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user (X).")
                break

            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            cv2.imshow(window_name, frame)

            now = time.time()

            if now >= next_capture_time and self.frame_count < self.MAX_FRAMES:
                ts_ms = int(time.time() * 1000)
                filename = os.path.join(self.output_dir, f"{ts_ms}.png")


                if round > 0:
                    self.save_queue.put((filename, frame.copy()))
                round += 1 #ignore the first frame
                self.frame_count += 1
                next_capture_time += self.FRAME_INTERVAL

            if (now - start_time) >= record_time or self.frame_count >= self.MAX_FRAMES:
                print(f"Stop: duration={now - start_time:.3f}s, frames={self.frame_count}")

                # 等待保存完成
                self.save_queue.join()

                # 停止保存线程
                self.save_thread_running = False
                self.save_queue.put(None)
                self.save_thread.join(timeout=2)

                cv2.destroyWindow(window_name)
                self.is_window_created = False
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Exit by key")
                self.save_thread_running = False
                self.save_queue.put(None)
                cv2.destroyWindow(window_name)
                self.is_window_created = False
                break

    def measure(self):
        """"""
        if not self.measurer:
            print("fail to initialize")
            return

        print("Start to measure\n")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self.is_window_created = True

        while True:
            #
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\ncolse")
                break

            ret, frame = self.cap.read()
            if not ret:
                print("no frame exists")
                break

            # begin to measure the distance
            measurement = self.measurer.measure_distance(frame)
            # begin to count the pixels
            pixel_info = self.pixel_counter.count_face_pixels(frame)
            if measurement:
                distance = measurement['distance_cm']
                pitch = measurement['pitch_degrees']
                yaw = measurement['yaw_degrees']
                roll = measurement['roll_degrees']
                pos_h = measurement['position_azimuth']  #
                pos_v = measurement['position_elevation']

                pixel_count = pixel_info['total_pixels'] if pixel_info else 0
                face_ratio = pixel_info['face_ratio_percent'] if pixel_info else 0

                print(f"\rDistance {distance:6.1f} cm | pitch: {pitch:+6.1f}° | yaw: {yaw:+6.1f}° "
                      f"| roll: {roll:+6.1f}°|angle: {pos_h}",
                      end='', flush=True)
            else:
                print("\rCan not detect face", end='\n', flush=True)

            #
            frame_display = self.measurer.draw_on_frame(frame, measurement) if self.measurer else frame
            frame_display = self.pixel_counter.draw_pixel_info(frame_display, pixel_info,
                                                               show_contour=True,
                                                               show_bbox=False)

            cv2.imshow(window_name, frame_display)
            #
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\nexit")
                break

        cv2.destroyAllWindows()
        self.is_window_created = False
        print("measurement ends")

    def __del__(self):
        if hasattr(self, 'save_thread') and self.save_thread and self.save_thread.is_alive():
            self.save_thread_running = False
            self.save_queue.put(None)
            self.save_thread.join(timeout=1)

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Camera released.")

#Todo time alignment issues