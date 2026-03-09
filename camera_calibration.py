"""
Camera Calibration Tool - Complete Version with Live Preview
For obtaining accurate camera intrinsic parameters (focal length, principal point, distortion coefficients)
Compatible with any camera, including virtual cameras (like Camo Studio)
"""

import cv2
import numpy as np
import os
from pathlib import Path
from config import data_settings as settings

class CameraCalibrationTool:
    """
    Camera Calibration Tool - Using Checkerboard Method

    Principle:
    By capturing the checkerboard pattern from different positions and angles,
    the OpenCV calibration algorithm calculates the camera's intrinsic matrix
    and distortion coefficients.
    """

    def __init__(self, checkerboard_size=(9, 6), square_size_mm=24.0):
        """
        Initialize the Calibration Tool

        Args:
            checkerboard_size: Number of inner corners (width, height)
            square_size_mm: Size of one square edge in millimeters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm

        # 3D points in real world space
        self.objp = np.zeros(
            (checkerboard_size[0] * checkerboard_size[1], 3),
            np.float32
        )
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0],
                                     0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_mm  # Convert to mm

        # Arrays to store object points and image points
        self.objpoints = []  # 3D point in real world space
        self.imgpoints = []  # 2D points in image plane
        self.image_size = None

        print("âœ“ Calibration Tool Initialized")
        print(f"  Pattern Size (Corners): {checkerboard_size[0]}x{checkerboard_size[1]}")
        print(f"  Square Size: {square_size_mm}mm")

    def collect_calibration_images(self, camera_index=0, num_images=20, delay_ms=500):
        """
        Real-time image collection for calibration with LIVE PREVIEW

        Args:
            camera_index: Camera device index (0=default)
            num_images: Target number of valid images to collect
            delay_ms: Delay after successful capture (ms)

        Controls:
            - Press SPACE to capture
            - Green Box = Pattern Detected âœ“
            - Red Box   = Pattern Not Detected âœ—
            - Press Q or ESC to quit
        """
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("âŒ Error: Could not open camera.")
            return False

        # Set high resolution for better precision
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Create save directory
        save_dir = Path("./calibration_images")
        save_dir.mkdir(exist_ok=True)

        print("\n" + "="*70)
        print("STARTING IMAGE COLLECTION WITH LIVE PREVIEW")
        print("="*70)
        print(f"\nTarget: {num_images} images")
        print("\n[Controls]")
        print("  1. Press SPACE to capture")
        print("  2. Green Box = Pattern Detected âœ“")
        print("  3. Red Box   = Pattern Not Detected âœ—")
        print("  4. Press Q or ESC to quit")
        print("\n[Tips]")
        print("  â€¢ Capture from different distances (Near, Mid, Far)")
        print("  â€¢ Capture different angles (Left, Right, Up, Down)")
        print("  â€¢ Ensure even lighting")
        print("  â€¢ Keep checkerboard at 30-45 degree angle")
        print("\n" + "-"*70 + "\n")

        count = 0
        successful_count = 0

        while successful_count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_detect, corners = cv2.findChessboardCorners(
                gray,
                self.checkerboard_size,
                None
            )

            # Display setup
            status_text = f"Progress: {successful_count}/{num_images}"
            frame_display = frame.copy()

            if ret_detect:
                # Green box for success
                cv2.rectangle(frame_display, (10, 10),
                            (frame.shape[1]-10, frame.shape[0]-10),
                            (0, 255, 0), 3)
                cv2.putText(frame_display, "âœ“ Pattern Detected", (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # Refine corner locations (Sub-pixel accuracy)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Draw corners
                cv2.drawChessboardCorners(frame_display, self.checkerboard_size, corners, ret_detect)
            else:
                # Red box for failure
                cv2.rectangle(frame_display, (10, 10),
                            (frame.shape[1]-10, frame.shape[0]-10),
                            (0, 0, 255), 3)
                cv2.putText(frame_display, "âœ— No Pattern Found", (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Add status info
            cv2.putText(frame_display, status_text,
                       (30, frame.shape[0]-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Add instructions at the bottom
            cv2.putText(frame_display, "SPACE=Capture, Q=Quit",
                       (frame.shape[1]-400, frame.shape[0]-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Display live preview window
            cv2.imshow("Calibration Capture - Live Preview", frame_display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # SPACE key
                if ret_detect:
                    # Save image and points
                    filename = save_dir / f"calibration_{successful_count:03d}.png"
                    cv2.imwrite(str(filename), frame)

                    self.objpoints.append(self.objp)
                    self.imgpoints.append(corners)
                    self.image_size = gray.shape[::-1]

                    successful_count += 1

                    print(f"âœ“ Image Saved: {successful_count}/{num_images}")

                    # Delay to prevent double clicks
                    cv2.waitKey(delay_ms)
                else:
                    print("âš  Cannot save: Pattern not detected. Adjust position and try again.")

            elif key == ord('q') or key == 27:  # Q or ESC
                print("\nCapture stopped by user.")
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nCollection Finished: {successful_count} valid images.")
        if successful_count >= 10:
            print(" Sufficient images collected for calibration")
            return True
        else:
            print(f" Warning: Low image count ({successful_count} < 10).")
            print("  Recommendation: Collect at least 10 valid images for accurate results")
            return False

    def calibrate(self):
        """
        Run calibration calculation

        Returns:
            (mtx, dist, rvecs, tvecs, reprojection_error) or None
        """
        if len(self.objpoints) < 3:
            print("Error: Not enough images (minimum 3 required).")
            return None

        print(f"\nRunning calibration on {len(self.objpoints)} images...")

        # Execute Calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            self.image_size,
            None,
            None
        )

        if not ret:
            print("Calibration Failed")
            return None

        # Calculate reprojection error
        reprojection_error = self._compute_reprojection_error(
            mtx, dist, rvecs, tvecs
        )

        return mtx, dist, rvecs, tvecs, reprojection_error

    def _compute_reprojection_error(self, mtx, dist, rvecs, tvecs):
        """Compute Reprojection Error (lower is better)"""
        total_error = 0.0
        total_points = 0

        for i in range(len(self.objpoints)):
            projected_points, _ = cv2.projectPoints(
                self.objpoints[i],
                rvecs[i],
                tvecs[i],
                mtx,
                dist
            )
            error = cv2.norm(self.imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
            total_error += error
            total_points += 1

        return total_error / total_points

    def print_calibration_results(self, mtx, dist, reprojection_error):
        """
        Print readable calibration results

        Args:
            mtx: Camera matrix
            dist: Distortion coefficients
            reprojection_error: Reprojection error
        """
        print("\n" + "="*70)
        print("CALIBRATION RESULTS")
        print("="*70)

        fx = mtx[0, 0]
        fy = mtx[1, 1]
        cx = mtx[0, 2]
        cy = mtx[1, 2]

        print(f"\n[Camera Intrinsic Matrix]")
        print(f"  Focal Length (fx): {fx:.2f} px")
        print(f"  Focal Length (fy): {fy:.2f} px")
        print(f"  Mean Focal Length: {(fx + fy) / 2:.2f} px")
        print(f"\n[Optical Center (Principal Point)]")
        print(f"  Center X (cx): {cx:.2f} px")
        print(f"  Center Y (cy): {cy:.2f} px")
        print(f"\n[Distortion Coefficients]")
        # Flatten dist array to handle both (5,) and (5,1) shapes
        dist_flat = dist.flatten()
        print(f"  k1 (Radial 1): {dist_flat[0]:.6f}")
        print(f"  k2 (Radial 2): {dist_flat[1]:.6f}")
        print(f"  p1 (Tangential 1): {dist_flat[2]:.6f}")
        print(f"  p2 (Tangential 2): {dist_flat[3]:.6f}")
        print(f"  k3 (Radial 3): {dist_flat[4]:.6f}")
        print(f"\n[Precision Metric]")
        print(f"  Reprojection Error: {reprojection_error:.4f} px")

        if reprojection_error < 0.5:
            print(f"   Excellent ( error < 0.5 px)")
        elif reprojection_error < 1.0:
            print(f"  Good (<1.0 px)")
        elif reprojection_error < 2.0:
            print(f"  Acceptable (<2.0 px)")
        else:
            print(f"  Suck - Consider recalibrating")

        print("\n" + "="*70)

    def save_calibration(self, mtx, dist, filename = "camera_calibration_{}.npz".format(settings["camera_name"])):
        """
        Save calibration results to file

        Args:
            mtx: Camera matrix
            dist: Distortion coefficients
            filename: Output filename
        """
        np.savez(
            filename,
            camera_matrix=mtx,
            dist_coeffs=dist,
            image_width=self.image_size[0],
            image_height=self.image_size[1]
        )
        print(f"\nâœ“ Calibration saved to: {filename}")

    def load_calibration(self, filename = "camera_calibration.npz"):
        """
        Load calibration results from file

        Returns:
            (mtx, dist, width, height) or None
        """
        try:
            data = np.load(filename)
            mtx = data['camera_matrix']
            dist = data['dist_coeffs']
            width = int(data['image_width'])
            height = int(data['image_height'])
            print(f" Calibration loaded from: {filename}")
            return mtx, dist, width, height
        except Exception as e:
            print(f"âŒ Failed to load: {e}")
            return None

    def generate_calibration_pattern(self, filename="checkerboard_pattern.png"):
        """
        Generate checkerboard pattern for printing

        Args:
            filename: Output filename
        """
        # Create 9x6 checkerboard (8x5 internal corners, 10x7 squares)
        square_size = 100
        width = 9 * square_size
        height = 6 * square_size

        checkerboard = np.zeros((height, width), dtype=np.uint8)

        for i in range(6):
            for j in range(9):
                if (i + j) % 2 == 0:
                    x1, y1 = j * square_size, i * square_size
                    x2, y2 = (j + 1) * square_size, (i + 1) * square_size
                    checkerboard[y1:y2, x1:x2] = 255

        cv2.imwrite(filename, checkerboard)
        print(f"Pattern generated: {filename}")
        print(f"  Size: {width}x{height} pixels")
        print(f"  For A4 printing: 10Ã—7 squares, 24mm each square")


def main():
    """
    Complete calibration workflow with live preview
    """
    print("\n" + "="*70)
    print("Camera Calibration Tool - Complete Workflow")
    print("="*70)

    # Initialize tool
    tool = CameraCalibrationTool(checkerboard_size=(9, 6), square_size_mm=24.0)

    # Step 1: Generate pattern
    print("\n[Step 1] Generate Calibration Pattern")
    print("-" * 70)
    tool.generate_calibration_pattern()
    print(">> Print 'checkerboard_pattern.png' (A4 landscape, 24mm squares)\n")

    # Step 2: Collect images with LIVE PREVIEW
    print("[Step 2] Collect Images with Live Preview")
    print("-" * 70)
    input(">> Ready to capture? Press ENTER to start...")
    success = tool.collect_calibration_images(camera_index=2, num_images=20)

    if not success:
        print("\nInsufficient images collected. Please try again.")
        return

    # Step 3: Calibrate
    print("\n[Step 3] Run Calibration")
    print("-" * 70)
    result = tool.calibrate()

    if result is None:
        print("Calibration failed")
        return

    mtx, dist, rvecs, tvecs, error = result

    # Step 4: Display results
    tool.print_calibration_results(mtx, dist, error)

    # Step 5: Save results
    print("\n[Step 5] Save Calibration")
    print("-" * 70)
    tool.save_calibration(mtx, dist)

    print("\n Calibration Complete!")
    print("You can now use 'camera_calibration.npz' for accurate distance measurements")


if __name__ == "__main__":
    main()