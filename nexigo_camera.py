import cv2
import time
import os
import threading
from queue import Queue

window_name = "NEXIGO Preview"


class Camera:
    def __init__(self, camera_index):
        self.frame_count = 0
        self.output_dir = None
        self.TARGET_FPS = 50.0
        self.FRAME_INTERVAL = 1.0 / self.TARGET_FPS  # 0.02s
        self.RECORD_DURATION = 10.0  # 10 秒
        self.MAX_FRAMES = int(self.TARGET_FPS * self.RECORD_DURATION)

        # 多线程保存队列
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

    def _save_worker(self):
        """后台线程：从队列中取图片并保存"""
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


                self.save_queue.put((filename, frame.copy()))

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

    def __del__(self):
        if hasattr(self, 'save_thread') and self.save_thread and self.save_thread.is_alive():
            self.save_thread_running = False
            self.save_queue.put(None)
            self.save_thread.join(timeout=1)

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Camera released.")

#Todo time alignment issues