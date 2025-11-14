import cv2
import time
import os

cam_index = 0
window_name = "NEXIGO Preview"

class Camera:
    def __init__(self):
        self.frame_count = 0
        self.output_dir = None
        self.TARGET_FPS = 50.0
        self.FRAME_INTERVAL = 1.0 / self.TARGET_FPS  # 0.02s
        self.RECORD_DURATION = 10.0  # 10 秒
        self.MAX_FRAMES = int(self.TARGET_FPS * self.RECORD_DURATION)
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.is_window_created = False
        if not self.cap.isOpened():
            print("Still cannot open camera, check if other apps are using it.")
            exit()
        self.cap.set(cv2.CAP_PROP_FPS, self.TARGET_FPS)
        print("Camera reported FPS:", self.cap.get(cv2.CAP_PROP_FPS))

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
            # 键盘退出：q 或 ESC
            if key == ord('q') or key == 27:
                print("Exit by key")
                break

    def record(self, record_time=10):
        # if run() has created window, we do not need to create window here
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
            # 如果用户点 X，并且窗口真关闭了，就退出
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user (X).")
                break

            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            cv2.imshow(window_name, frame)

            now = time.time()

            # 按时间间隔采样
            if now >= next_capture_time and self.frame_count < self.MAX_FRAMES:
                ts_ms = int(time.time() * 1000)
                filename = os.path.join(self.output_dir, f"{ts_ms}.png")
                cv2.imwrite(filename, frame)

                self.frame_count += 1
                next_capture_time += self.FRAME_INTERVAL

            # 达到录制时长或帧数限制退出
            if (now - start_time) >= record_time or self.frame_count >= self.MAX_FRAMES:
                print(f"Stop: duration={now - start_time:.3f}s, frames={self.frame_count}")
                self.is_window_created = False
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Exit by key")
                break

    # ---- 创建 ./data/video/session_xxxxx 文件夹 ----
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Camera released.")


#camera = Camera()
#camera.record()

