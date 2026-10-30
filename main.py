"""AURA Phase 1: webcam face mesh, head pose, calibration, and FPS."""
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
from aura.core.visuals import draw_mesh, draw_panel

ROOT = Path(__file__).resolve().parent


def run(camera_index: int, width: int, height: int, smoothing: float, mirror: bool) -> None:
    model_path = ensure_model(ROOT / "models" / "face_landmarker.task")
    # Windows: DirectShow is generally good for webcams; fall back to the default backend.
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}. Try --camera 1, close video-call apps, and check camera permissions.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )

    smoother = PoseSmoother(alpha=smoothing)
    neutral = np.zeros(3, dtype=np.float64)
    raw_pose = None
    smoothed_pose = None
    show_mesh = True
    fps = 0.0
    last_frame_time = time.perf_counter()
    last_timestamp_ms = -1
    start = time.monotonic()

    print("AURA Phase 1 | C or SPACE = calibrate | M = toggle mesh | R = reset | Q or ESC = quit")
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    print("Camera stopped supplying frames.")
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)

                # Timestamps must increase strictly for MediaPipe VIDEO mode.
                timestamp_ms = max(int((time.monotonic() - start) * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                result = detector.detect_for_video(mp_image, timestamp_ms)

                detected = bool(result.face_landmarks and result.facial_transformation_matrixes)
                if detected:
                    # Rotation is derived from the Face Landmarker transformation matrix.
                    transform = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64).reshape(4, 4)
                    raw_pose = euler_from_matrix(transform[:3, :3])
                    smoothed_pose = smoother.update(raw_pose)
                    if show_mesh:
                        draw_mesh(frame, result.face_landmarks[0])
                else:
                    raw_pose = None
                    smoothed_pose = None
                    smoother.reset()

                now = time.perf_counter()
                delta = now - last_frame_time
                last_frame_time = now
                if delta > 0:
                    instant = min(1.0 / delta, 240.0)
                    fps = instant if fps == 0 else 0.85 * fps + 0.15 * instant

                relative_pose = smoothed_pose - neutral if smoothed_pose is not None else None
                draw_panel(frame, relative_pose, fps, detected, show_mesh, smoothing)
                cv2.imshow("AURA | Phase 1 - Tracking Foundation", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key in (ord("c"), 32) and smoothed_pose is not None:
                    neutral = smoothed_pose.copy()
                    print("Neutral pose calibrated. Look forward and press C again to recalibrate.")
                elif key == ord("m"):
                    show_mesh = not show_mesh
                elif key == ord("r"):
                    neutral[:] = 0
                    smoother.reset()
                    print("Calibration and smoothing reset.")
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="AURA Phase 1 head tracking MVP")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--smoothing", type=float, default=0.28, help="EMA alpha: 0 < alpha <= 1 (lower is smoother)")
    parser.add_argument("--no-mirror", action="store_true", help="display unmirrored webcam video")
    args = parser.parse_args()
    if not 0 < args.smoothing <= 1:
        parser.error("--smoothing must be greater than 0 and at most 1")
    try:
        run(args.camera, args.width, args.height, args.smoothing, not args.no_mirror)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
