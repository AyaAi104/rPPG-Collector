"""
Face Pixel Counter Module

This module calculates the total number of pixels within the detected face region.
It uses MediaPipe Face Mesh face-oval landmarks to build a polygon mask and count
the pixels inside that region.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple
import mediapipe as mp


class FacePixelCounter:
    """
    Count the total number of pixels within the face region.

    Steps:
    1. Use MediaPipe Face Mesh to obtain face landmarks.
    2. Build a convex hull / polygon mask of the face region based on face-oval points.
    3. Count the number of pixels inside the mask.
    """

    # MediaPipe Face Mesh face-oval landmark indices
    FACE_OVAL_INDICES = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]

    def __init__(self):
        """Initialize the face pixel counter."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        print("✓ Face Pixel Counter initialized")

    def count_face_pixels(self, frame: np.ndarray,
                          face_landmarks=None) -> Optional[Dict]:
        """
        Count the total number of pixels inside the face region.

        Args:
            frame: Input image frame (BGR).
            face_landmarks: Optional pre-detected face landmarks (from an external detector).

        Returns:
            A dictionary with pixel statistics, or None if no face is detected.
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]

        # If no landmarks are provided, detect them using Face Mesh
        if face_landmarks is None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if not results.multi_face_landmarks:
                return None

            face_landmarks = results.multi_face_landmarks[0]

        # Extract face-oval points
        face_oval_points = []
        for idx in self.FACE_OVAL_INDICES:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                # Boundary check
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                face_oval_points.append([x, y])

        if len(face_oval_points) < 3:
            return None

        face_oval_points = np.array(face_oval_points, dtype=np.int32)

        # Create a binary mask for the face region
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_oval_points], 255)

        # Count the number of face pixels
        total_pixels = cv2.countNonZero(mask)

        # Compute bounding box of the face polygon
        x_min, y_min = face_oval_points.min(axis=0)
        x_max, y_max = face_oval_points.max(axis=0)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        bbox_area = bbox_width * bbox_height

        # Compute ratio of face pixels to total frame pixels
        frame_total_pixels = h * w
        face_ratio = (total_pixels / frame_total_pixels) * 100

        return {
            'total_pixels': total_pixels,
            'bbox': (x_min, y_min, x_max, y_max),
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'bbox_area': bbox_area,
            'face_ratio_percent': face_ratio,
            'face_contour_points': face_oval_points,
            'mask': mask
        }

    def draw_pixel_info(self, frame: np.ndarray,
                        pixel_info: Optional[Dict],
                        show_contour: bool = True,
                        show_bbox: bool = False) -> np.ndarray:
        """
        Draw pixel information on the frame.

        Args:
            frame: Input image frame (BGR).
            pixel_info: Dictionary returned by count_face_pixels.
            show_contour: Whether to draw the face contour polygon.
            show_bbox: Whether to draw the bounding box.

        Returns:
            The annotated image frame.
        """
        if frame is None or frame.size == 0:
            return frame

        if pixel_info is None:
            cv2.putText(frame, "Face Pixels: N/A", (30, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            return frame

        # Draw face contour polygon
        if show_contour and 'face_contour_points' in pixel_info:
            cv2.polylines(frame, [pixel_info['face_contour_points']],
                          True, (255, 255, 0), 2)

        # Draw bounding box
        if show_bbox and 'bbox' in pixel_info:
            x_min, y_min, x_max, y_max = pixel_info['bbox']
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                          (0, 255, 255), 2)

        # Draw pixel statistics text
        total_pixels = pixel_info['total_pixels']
        face_ratio = pixel_info['face_ratio_percent']

        # Formatted text (with thousand separators)
        pixel_text = f"Face Pixels: {total_pixels:,}"
        ratio_text = f"Face Ratio: {face_ratio:.2f}%"

        cv2.putText(frame, pixel_text, (30, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, ratio_text, (30, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame

    def __del__(self):
        """Release resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


# Standalone test
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    counter = FacePixelCounter()

    print("Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pixel_info = counter.count_face_pixels(frame)
        frame = counter.draw_pixel_info(frame, pixel_info)

        cv2.imshow("Face Pixel Counter Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
