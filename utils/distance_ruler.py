"""
Accurate Face Distance Measurement System - Refined Version

Usage:
This script combines stereo vision principles with MediaPipe's face landmarks and
camera calibration to accurately measure the distance to a person's face.

Improvements:
- Attempts to automatically read camera focal length.
- Includes bug fixes and enhanced robustness.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CameraCalibration:
    """
    Stores camera calibration parameters.
    """
    focal_length: float  # Focal length in pixels
    principal_point: Tuple[float, float]  # Principal point coordinates (cx, cy)
    image_width: int
    image_height: int

    @staticmethod
    def from_intrinsics(fx: float, fy: float, cx: float, cy: float,
                        img_w: int, img_h: int):
        """
        Creates calibration parameters from an intrinsic matrix.

        Args:
            fx (float): Focal length along the x-axis.
            fy (float): Focal length along the y-axis.
            cx (float): Principal point x-coordinate.
            cy (float): Principal point y-coordinate.
            img_w (int): Image width.
            img_h (int): Image height.

        Returns:
            CameraCalibration: An instance of the class.
        """
        # Use the average of fx and fy as the focal length
        focal_length = (fx + fy) / 2
        return CameraCalibration(focal_length, (cx, cy), img_w, img_h)

    @staticmethod
    def from_fov(fov_degrees: float, image_width: int, image_height: int):
        """
        Estimates focal length from the field of view (FOV).

        Args:
            fov_degrees (float): The horizontal field of view in degrees.
            image_width (int): The width of the image.
            image_height (int): The height of the image.

        Returns:
            CameraCalibration: An instance of the class.
        """
        fov_rad = np.radians(fov_degrees)
        focal_length = image_width / (2 * np.tan(fov_rad / 2))
        return CameraCalibration(
            focal_length=focal_length,
            principal_point=(image_width / 2, image_height / 2),
            image_width=image_width,
            image_height=image_height
        )


class FaceDistanceMeasurement:
    """
    A precise face distance estimator.

    Methodology:
    1. Detects multiple key facial landmarks (eyes, nose, mouth corners).
    2. Uses the PnP (Perspective-n-Point) algorithm to estimate the 3D pose of the face.
    3. Applies a smoothing filter to the results to reduce jitter and noise.
    """

    # 3D model points of a standard face (in millimeters).
    # Based on average adult facial dimensions.
    FACE_MODEL_3D = np.array([
        [0.0, 0.0, 0.0],           # Nose tip (origin)
        [-30.0, -65.0, -20.0],     # Left eye
        [30.0, -65.0, -20.0],      # Right eye
        [-60.0, 10.0, -30.0],      # Left mouth corner
        [60.0, 10.0, -30.0],       # Right mouth corner
        [0.0, 70.0, -30.0],        # Chin
        [0.0, -35.0, -40.0],       # Forehead center
    ], dtype=np.float32)

    # Corresponding indices in MediaPipe's face landmarks.
    MEDIAPIPE_INDICES = [
        4,      # Nose tip
        130,    # Left eye
        359,    # Right eye
        308,    # Left mouth corner
        78,     # Right mouth corner
        152,    # Chin
        10,     # Forehead
    ]

    def __init__(self, calibration: CameraCalibration):
        """
        Initializes the distance estimator.

        Args:
            calibration (CameraCalibration): The camera calibration parameters.
        """
        self.calibration = calibration

        # Initialize MediaPipe Face Mesh.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # Refine landmarks for more accuracy.
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Camera matrix - BUG FIX: Correctly set focal length from calibration.
        self.camera_matrix = np.array([
            [calibration.focal_length, 0, calibration.principal_point[0]],
            [0, calibration.focal_length, calibration.principal_point[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Distortion coefficients (assuming no distortion, can be adjusted).
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        # Filter parameters using deques for efficient fixed-size buffering.
        self.distance_buffer = deque(maxlen=10)
        self.angle_buffer_x = deque(maxlen=10)
        self.angle_buffer_y = deque(maxlen=10)
        self.angle_buffer_z = deque(maxlen=10)

        print(f"✓ Distance estimator initialized successfully.")
        print(f"  Focal Length: {calibration.focal_length:.2f} pixels")
        print(f"  Principal Point: ({calibration.principal_point[0]:.1f}, {calibration.principal_point[1]:.1f})")
        print(f"  Resolution: {calibration.image_width}x{calibration.image_height}")

    def measure_distance(self, frame: np.ndarray) -> Optional[dict]:
        """
        Measures the distance from the camera to a face in the frame.

        Args:
            frame (np.ndarray): The input video frame.

        Returns:
            A dictionary containing distance and angle information,
            or None if no face is detected.
        """
        # BUG FIX: Check if frame is None or empty.
        if frame is None or frame.size == 0:
            return None

        h, w, c = frame.shape

        # Convert the BGR image to RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find face landmarks.
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]

        # Extract the 2D coordinates of the key landmarks.
        face_2d = []
        for idx in self.MEDIAPIPE_INDICES:
            # BUG FIX: Add boundary check for landmark index.
            if idx >= len(landmarks.landmark):
                return None
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            # BUG FIX: Check for valid coordinates.
            if x < 0 or x >= w or y < 0 or y >= h:
                return None
            face_2d.append([x, y])

        face_2d = np.array(face_2d, dtype=np.float32)

        # BUG FIX: Ensure there are enough points for PnP.
        if len(face_2d) < 4:
            return None

        # Solve for the 3D pose of the face using the PnP algorithm.
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.FACE_MODEL_3D,
            face_2d,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP  # EPNP method is fast and accurate.
        )

        if not success or rotation_vec is None or translation_vec is None:
            return None

        # BUG FIX: Check if the translation vector is valid (z > 0).
        if translation_vec[2][0] <= 0:
            return None

        # Extract the distance (z-axis, in millimeters).
        distance_mm = translation_vec[2][0]
        distance_cm = distance_mm / 10.0

        x_mm = translation_vec[0][0]
        y_mm = translation_vec[1][0]
        # Azimuth we need
        azimuth_rad = np.arctan2(x_mm, distance_mm)
        #Elevation tan(phi) = y / z
        elevation_rad = np.arctan2(y_mm, distance_mm)

        pos_azimuth = np.degrees(azimuth_rad)
        pos_elevation = np.degrees(elevation_rad)
        # Convert rotation vector to Euler angles.
        try:
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            #angles = self._rotation_matrix_to_euler_angles(rotation_mat)
            angles = cv2.RQDecomp3x3(rotation_mat)[0]
        except:
            return None

        # Apply a moving average filter.
        self.distance_buffer.append(distance_cm)
        self.angle_buffer_x.append(angles[0])
        self.angle_buffer_y.append(angles[1])
        self.angle_buffer_z.append(angles[2])

        distance_filtered = np.mean(self.distance_buffer) if self.distance_buffer else distance_cm
        angle_x_filtered = np.mean(self.angle_buffer_x) if self.angle_buffer_x else angles[0]
        angle_y_filtered = np.mean(self.angle_buffer_y) if self.angle_buffer_y else angles[1]
        angle_z_filtered = np.mean(self.angle_buffer_z) if self.angle_buffer_z else angles[2]

        return {
            'distance_cm': distance_filtered,
            'distance_m': distance_filtered / 100.0,
            'distance_mm': distance_filtered * 10.0,
            #'pitch_degrees': np.degrees(angle_x_filtered),  # Up-down angle
            #'yaw_degrees': np.degrees(angle_y_filtered),    # Left-right angle
            #'roll_degrees': np.degrees(angle_z_filtered),   # Tilt angle
            'pitch_degrees': (angle_x_filtered),  # Up-down angle
            'yaw_degrees': (angle_y_filtered),  # Left-right angle
            'roll_degrees': (angle_z_filtered),  # Tilt angle
            'face_2d_points': face_2d,
            'rotation_vec': rotation_vec,
            'translation_vec': translation_vec,
            'raw_distance_cm': distance_cm,  # Unfiltered raw value
            'position_azimuth': pos_azimuth,  # 水平偏角 (正值通常在画面右侧)
            'position_elevation': pos_elevation, # 垂直偏角 (正值通常在画面下方)
            'distance_x': x_mm/10
        }

    @staticmethod
    def _rotation_matrix_to_euler_angles(rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Converts a rotation matrix to Euler angles.

        Returns:
            np.ndarray: [pitch, yaw, roll] in radians.
        """
        # BUG FIX: Add numerical stability check.
        sin_pitch = -rotation_matrix[2, 0]
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)  # Prevent arcsin domain error.

        pitch = np.arcsin(sin_pitch)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

        return np.array([pitch, yaw, roll])

    def draw_on_frame(self, frame: np.ndarray, measurement: dict) -> np.ndarray:
        """
        Draws measurement results onto the video frame.

        Args:
            frame (np.ndarray): The input video frame.
            measurement (dict): The dictionary of measurement results.

        Returns:
            np.ndarray: The annotated video frame.
        """
        # BUG FIX: Check for valid frame and measurement data.
        if frame is None or frame.size == 0:
            return frame

        if measurement is None:
            cv2.putText(frame, "No face detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return frame

        h, w = frame.shape[:2]

        # Draw the 2D facial landmarks.
        for point in measurement['face_2d_points']:
            pt = tuple(point.astype(int))
            # BUG FIX: Check for valid coordinates before drawing.
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(frame, pt, 3, (0, 255, 0), -1)

        # Draw the distance information.
        distance_text = f"Distance: {measurement['distance_cm']:.1f} cm"
        cv2.putText(frame, distance_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Draw the angle information.
        pitch_text = f"Pitch: {measurement['pitch_degrees']:.1f} deg"
        yaw_text = f"Yaw: {measurement['yaw_degrees']:.1f} deg"
        roll_text = f"Roll: {measurement['roll_degrees']:.1f} deg"
        angle_text = f"Angle: {measurement['position_azimuth']:.1f} deg"
        distance_x = f"x: { measurement['distance_x']:.1f} cm"

        cv2.putText(frame, pitch_text, (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, yaw_text, (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, roll_text, (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, angle_text, (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, distance_x, (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Draw the 3D coordinate axes.
        try:
            self._draw_3d_axes(frame, measurement)
        except Exception as e:
            # If drawing fails for any reason, continue without crashing.
            # print(f"Could not draw 3D axes: {e}")
            pass

        return frame

    def _draw_3d_axes(self, frame: np.ndarray, measurement: dict, axis_length: float = 50):
        """Draws the 3D coordinate axes on the face."""
        rotation_vec = measurement['rotation_vec']
        translation_vec = measurement['translation_vec']

        # 3D endpoints of the axes.
        axis_points_3d = np.float32([
            [0, 0, 0],           # Origin
            [axis_length, 0, 0], # X-axis (Red)
            [0, axis_length, 0], # Y-axis (Green)
            [0, 0, axis_length]  # Z-axis (Blue)
        ])

        # Project the 3D points to the 2D image plane.
        axis_points_2d, _ = cv2.projectPoints(
            axis_points_3d,
            rotation_vec,
            translation_vec,
            self.camera_matrix,
            self.dist_coeffs
        )

        axis_points_2d = axis_points_2d.astype(int)
        origin = tuple(axis_points_2d[0].ravel())

        # Draw the axis lines.
        cv2.line(frame, origin, tuple(axis_points_2d[1].ravel()), (0, 0, 255), 3)  # X - Red
        cv2.line(frame, origin, tuple(axis_points_2d[2].ravel()), (0, 255, 0), 3)  # Y - Green
        cv2.line(frame, origin, tuple(axis_points_2d[3].ravel()), (255, 0, 0), 3)  # Z - Blue


def get_camera_focal_length(cap: cv2.VideoCapture) -> Optional[float]:
    """
    Attempts to read the focal length directly from the camera properties.

    Note: Not all cameras support this feature via OpenCV properties.

    Args:
        cap (cv2.VideoCapture): The OpenCV VideoCapture object.

    Returns:
        The focal length in pixels, or None if it cannot be determined.
    """
    try:
        # These properties are not standard, but some cameras might expose them.
        # This is an experimental attempt.
        focal_x = cap.get(cv2.CAP_PROP_FOCAL_LENGTH)

        if focal_x > 0:
            print(f"✓ Successfully read focal length from camera: {focal_x:.2f} pixels")
            return focal_x
    except:
        pass

    return None


def create_optimal_calibration(cap: cv2.VideoCapture = None,
                              frame_width: int = 1280,
                              frame_height: int = 720,
                              fov_horizontal: float = 70.0) -> CameraCalibration:
    """
    Creates the best possible camera calibration parameters.

    Improvement: First, it tries to read focal length from the camera.
                 If that fails, it falls back to estimating from a default FOV.

    Args:
        cap (cv2.VideoCapture, optional): The VideoCapture object. Defaults to None.
        frame_width (int): The width of the video frame. Defaults to 1280.
        frame_height (int): The height of the video frame. Defaults to 720.
        fov_horizontal (float): The horizontal Field of View in degrees. Defaults to 70.0.

    Returns:
        CameraCalibration: The best available calibration parameters.
    """
    print("Creating camera calibration parameters...")

    # Try to read focal length directly from the camera.
    focal_length = None
    if cap is not None and cap.isOpened():
        focal_length = get_camera_focal_length(cap)

    # If reading fails, estimate from the Field of View.
    if focal_length is None:
        print(f" Could not read focal length from camera. Estimating from FOV ({fov_horizontal}°).")
        return CameraCalibration.from_fov(fov_horizontal, frame_width, frame_height)
    else:
        # If successfully read, use the measured value.
        return CameraCalibration(
            focal_length=focal_length,
            principal_point=(frame_width / 2, frame_height / 2),
            image_width=frame_width,
            image_height=frame_height
        )


if __name__ == "__main__":
    # Example usage: Measure distance from a live webcam feed.
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Create calibration parameters (attempts to auto-detect from camera).
    calibration = create_optimal_calibration(cap, FRAME_WIDTH, FRAME_HEIGHT)

    # Initialize the distance measurement tool.
    measurer = FaceDistanceMeasurement(calibration)

    print("\nStarting distance measurement (Press 'q' to exit)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Measure the distance.
        measurement = measurer.measure_distance(frame)

        # Draw the results on the frame.
        annotated_frame = measurer.draw_on_frame(frame, measurement)

        # Display the result.
        cv2.imshow("Face Distance Measurement", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()