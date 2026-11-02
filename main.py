
"""AURA v0.2 - Head rotation + relative 3D position tracking."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from aura.core.model import ensure_model
from aura.core.pose import PoseSmoother, euler_from_matrix
from aura.core.visuals import draw_dashboard, draw_mesh

ROOT = Path(__file__).resolve().parent
WINDOW_NAME = "AURA | Spatial Tracking v0.2"


class PositionTracker:
    """
    Estimates relative head movement from facial landmarks.

    X/Y: percent of camera frame from calibrated position.
    Z: percent change in apparent inter-eye distance.

    These are relative estimates, not metric 3D coordinates.
    """

    def __init__(self, alpha: float = 0.28):
        self.alpha = alpha
        self.smoothed = None
        self.neutral = None

    def update(self, landmarks, width: int, height: int):
        # Outer corners of the eyes (MediaPipe indices)
        left = landmarks[33]
        right = landmarks[263]

        # Eye midpoint as an approximate head anchor
        center_x = (left.x + right.x) / 2.0
        center_y = (left.y + right.y) / 2.0

        # Apparent eye separation in pixels.
        # Correct for image aspect ratio.
        dx = (right.x - left.x) * width
        dy = (right.y - left.y) * height
        eye_distance = float(np.hypot(dx, dy))

        value = np.array(
            [center_x, center_y, eye_distance],
            dtype=np.float64,
        )

        if self.smoothed is None:
            self.smoothed = value.copy()
        else:
            self.smoothed += self.alpha * (
                value - self.smoothed
            )

        if self.neutral is None:
            self.calibrate()

        return self.relative()

    def calibrate(self):
        if self.smoothed is not None:
            self.neutral = self.smoothed.copy()

    def relative(self):
        if self.smoothed is None or self.neutral is None:
            return None

        x = (self.smoothed[0] - self.neutral[0]) * 100.0

        # Negative sign makes upward movement positive
        y = (self.neutral[1] - self.smoothed[1]) * 100.0

        baseline_distance = max(self.neutral[2], 1.0)

        # Larger apparent face = closer to camera
        z = (
            (self.smoothed[2] / baseline_distance) - 1.0
        ) * 100.0

        return np.array([x, y, z], dtype=np.float64)

    def lose_tracking(self):
        self.smoothed = None

    def reset(self):
        self.smoothed = None
        self.neutral = None


def run(
    camera_index: int,
    width: int,
    height: int,
    smoothing: float,
    mirror: bool,
) -> None:

    model_path = ensure_model(
        ROOT / "models" / "face_landmarker.task"
    )

    backend = (
        cv2.CAP_DSHOW
        if sys.platform == "win32"
        else cv2.CAP_ANY
    )

    capture = cv2.VideoCapture(camera_index, backend)

    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open camera {camera_index}. "
            "Check Windows camera permissions."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )

    rotation_tracker = PoseSmoother(alpha=smoothing)
    position_tracker = PositionTracker(alpha=smoothing)

    neutral_rotation = np.zeros(3, dtype=np.float64)
    current_rotation = None
    current_position = None

    show_mesh = True
    show_gauges = True

    fps = 0.0
    last_frame_time = time.perf_counter()
    start_time = time.monotonic()
    last_timestamp_ms = -1

    print("\nAURA v0.2 - Spatial Tracking")
    print("C / SPACE : Calibrate")
    print("M         : Toggle face mesh")
    print("G         : Toggle gauges")
    print("R         : Reset calibration")
    print("Q / ESC   : Quit\n")

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )
    cv2.resizeWindow(WINDOW_NAME, 1240, 720)

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        ) as detector:

            while True:
                ok, frame = capture.read()

                if not ok:
                    print("Camera stopped providing frames.")
                    break

                if mirror:
                    frame = cv2.flip(frame, 1)

                frame_height, frame_width = frame.shape[:2]

                timestamp_ms = max(
                    int(
                        (time.monotonic() - start_time) * 1000
                    ),
                    last_timestamp_ms + 1
                )
                last_timestamp_ms = timestamp_ms

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb)
                )

                result = detector.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                detected = bool(
                    result.face_landmarks
                    and result.facial_transformation_matrixes
                )

                if detected:
                    landmarks = result.face_landmarks[0]

                    transform = np.asarray(
                        result.facial_transformation_matrixes[0],
                        dtype=np.float64
                    ).reshape(4, 4)

                    raw_rotation = euler_from_matrix(
                        transform[:3, :3]
                    )

                    current_rotation = rotation_tracker.update(
                        raw_rotation
                    )

                    current_position = position_tracker.update(
                        landmarks,
                        frame_width,
                        frame_height
                    )

                    if show_mesh:
                        draw_mesh(frame, landmarks)

                else:
                    current_rotation = None
                    current_position = None

                    rotation_tracker.reset()
                    position_tracker.lose_tracking()

                relative_rotation = (
                    current_rotation - neutral_rotation
                    if current_rotation is not None
                    else None
                )

                now = time.perf_counter()
                delta = now - last_frame_time
                last_frame_time = now

                if delta > 0:
                    instant_fps = min(1.0 / delta, 240.0)
                    fps = (
                        instant_fps
                        if fps == 0
                        else fps * 0.85 + instant_fps * 0.15
                    )

                dashboard = draw_dashboard(
                    frame=frame,
                    rotation=relative_rotation,
                    position=current_position,
                    fps=fps,
                    detected=detected,
                    show_mesh=show_mesh,
                    show_gauges=show_gauges,
                    smoothing=smoothing,
                )

                cv2.imshow(WINDOW_NAME, dashboard)

                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    break

                elif key in (ord("c"), 32):
                    if current_rotation is not None:
                        neutral_rotation = current_rotation.copy()
                        position_tracker.calibrate()
                        current_position = position_tracker.relative()
                        print("Position and rotation calibrated.")

                elif key == ord("m"):
                    show_mesh = not show_mesh
                    print(f"Face mesh: {show_mesh}")

                elif key == ord("g"):
                    show_gauges = not show_gauges
                    print(f"Gauges: {show_gauges}")

                elif key == ord("r"):
                    neutral_rotation[:] = 0
                    rotation_tracker.reset()
                    position_tracker.reset()
                    current_rotation = None
                    current_position = None
                    print("Tracking calibration reset.")

    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AURA v0.2 - Spatial Head Tracking"
    )

    parser.add_argument(
        "--camera", type=int, default=0
    )
    parser.add_argument(
        "--width", type=int, default=640
    )
    parser.add_argument(
        "--height", type=int, default=480
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.28
    )
    parser.add_argument(
        "--no-mirror", action="store_true"
    )

    args = parser.parse_args()

    if not 0 < args.smoothing <= 1:
        parser.error(
            "--smoothing must be between 0 (exclusive) and 1"
        )

    try:
        run(
            args.camera,
            args.width,
            args.height,
            args.smoothing,
            not args.no_mirror,
        )

    except (RuntimeError, OSError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
