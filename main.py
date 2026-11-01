
"""AURA Phase 1: webcam face mesh, head pose, calibration, axes, and FPS."""
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
from aura.core.visuals import draw_mesh, draw_panel, draw_axes_widget

ROOT = Path(__file__).resolve().parent


def run(
    camera_index: int,
    width: int,
    height: int,
    smoothing: float,
    mirror: bool,
) -> None:

    # Load MediaPipe face tracking model
    model_path = ensure_model(ROOT / "models" / "face_landmarker.task")

    # Initialize webcam
    capture = cv2.VideoCapture(
        camera_index,
        cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY,
    )

    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open camera {camera_index}. "
            "Try --camera 1, close video-call apps, "
            "and check camera permissions."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Configure MediaPipe Face Landmarker
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

    # Tracking state
    smoother = PoseSmoother(alpha=smoothing)

    neutral = np.zeros(3, dtype=np.float64)

    raw_pose = None
    smoothed_pose = None

    # Display toggles
    show_mesh = True
    show_axes = True

    # FPS tracking
    fps = 0.0
    last_frame_time = time.perf_counter()

    # MediaPipe timestamps
    last_timestamp_ms = -1
    start = time.monotonic()

    print(
        "AURA Phase 1 | "
        "C/SPACE = Calibrate | "
        "M = Toggle Mesh | "
        "G = Toggle Axes | "
        "R = Reset | "
        "Q/ESC = Quit"
    )

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        ) as detector:

            while True:

                # Capture frame
                ok, frame = capture.read()

                if not ok:
                    print("Camera stopped supplying frames.")
                    break

                # Mirror webcam
                if mirror:
                    frame = cv2.flip(frame, 1)

                # Create strictly increasing timestamp
                timestamp_ms = max(
                    int((time.monotonic() - start) * 1000),
                    last_timestamp_ms + 1,
                )

                last_timestamp_ms = timestamp_ms

                # Convert BGR -> RGB
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )

                # Run face tracking
                result = detector.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                detected = bool(
                    result.face_landmarks
                    and result.facial_transformation_matrixes
                )

                if detected:

                    # Extract head rotation matrix
                    transform = np.asarray(
                        result.facial_transformation_matrixes[0],
                        dtype=np.float64,
                    ).reshape(4, 4)

                    # Calculate yaw, pitch, roll
                    raw_pose = euler_from_matrix(
                        transform[:3, :3]
                    )

                    # Apply smoothing
                    smoothed_pose = smoother.update(raw_pose)

                    # Draw face landmarks
                    if show_mesh:
                        draw_mesh(
                            frame,
                            result.face_landmarks[0],
                        )

                else:
                    raw_pose = None
                    smoothed_pose = None
                    smoother.reset()

                # Calculate FPS
                now = time.perf_counter()
                delta = now - last_frame_time
                last_frame_time = now

                if delta > 0:
                    instant = min(1.0 / delta, 240.0)

                    fps = (
                        instant
                        if fps == 0
                        else 0.85 * fps + 0.15 * instant
                    )

                # Calculate pose relative to neutral calibration
                relative_pose = (
                    smoothed_pose - neutral
                    if smoothed_pose is not None
                    else None
                )

                # Draw information panel
                draw_panel(
                    frame,
                    relative_pose,
                    fps,
                    detected,
                    show_mesh,
                    show_axes,
                    smoothing,
                )

                # Draw X/Y and roll axis visualization
                draw_axes_widget(
                    frame,
                    relative_pose,
                    show_axes,
                )

                # Display webcam
                cv2.imshow(
                    "AURA | Phase 1 - Tracking Foundation",
                    frame,
                )

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                # Quit
                if key in (27, ord("q")):
                    break

                # Calibrate neutral pose
                if key in (ord("c"), 32):

                    if smoothed_pose is not None:
                        neutral = smoothed_pose.copy()

                        print(
                            "Neutral pose calibrated. "
                            "Look forward and press C again "
                            "to recalibrate."
                        )

                # Toggle face mesh
                elif key == ord("m"):
                    show_mesh = not show_mesh

                    print(
                        f"Face mesh: {'ON' if show_mesh else 'OFF'}"
                    )

                # Toggle head axes widget
                elif key == ord("g"):
                    show_axes = not show_axes

                    print(
                        f"Head axes: {'ON' if show_axes else 'OFF'}"
                    )

                # Reset calibration and smoothing
                elif key == ord("r"):
                    neutral[:] = 0
                    smoother.reset()

                    print(
                        "Calibration and smoothing reset."
                    )

    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:

    parser = argparse.ArgumentParser(
        description="AURA Phase 1 head tracking MVP"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default: 0)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.28,
        help="EMA alpha: 0 < alpha <= 1 (lower is smoother)",
    )

    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Display unmirrored webcam video",
    )

    args = parser.parse_args()

    if not 0 < args.smoothing <= 1:
        parser.error(
            "--smoothing must be greater than 0 and at most 1"
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
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
