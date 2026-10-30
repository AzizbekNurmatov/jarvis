"""Draw simple webcam diagnostics; no desktop HUD in Phase 1."""
from __future__ import annotations

import cv2
import numpy as np

# Index pairs sampled from the standard MediaPipe face mesh: eyes, brows,
# nose, lips, outer oval. A lightweight subset avoids costly full-mesh drawing.
CONTOURS = [
    [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,
     148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10],
    [33,160,158,133,153,144,33],
    [362,385,387,263,373,380,362],
    [61,40,37,0,267,270,291,321,314,17,84,91,61],
    [168,6,197,195,5,4,1,2,98],
]


def draw_mesh(frame: np.ndarray, landmarks) -> None:
    height, width = frame.shape[:2]
    # Sparse points + wire contours are easier to see than every triangulation.
    for contour in CONTOURS:
        points = []
        for idx in contour:
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            points.append((int(lm.x * width), int(lm.y * height)))
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, (255, 225, 75), 1, cv2.LINE_AA)
    for idx in range(0, min(len(landmarks), 468), 5):
        lm = landmarks[idx]
        x, y = int(lm.x * width), int(lm.y * height)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 1, (80, 220, 255), -1, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, pose: np.ndarray | None, fps: float,
               detected: bool, show_mesh: bool, smoothing: float) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    panel_width = min(width - 16, 355)
    cv2.rectangle(overlay, (8, 8), (panel_width, 174), (15, 24, 34), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cyan = (240, 216, 74)  # BGR
    white = (238, 238, 238)
    muted = (175, 186, 195)
    status_color = (100, 220, 100) if detected else (80, 90, 255)

    def line(text: str, y: int, color=white, scale=0.53) -> None:
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    line("AURA / PHASE 1", 29, cyan, 0.62)
    line("FACE FOUND" if detected else "NO FACE DETECTED", 52, status_color)
    if pose is not None:
        yaw, pitch, roll = pose
        line(f"YAW:   {yaw:+7.1f} deg", 78)
        line(f"PITCH: {pitch:+7.1f} deg", 99)
        line(f"ROLL:  {roll:+7.1f} deg", 120)
    else:
        line("YAW:     --", 78, muted)
        line("PITCH:   --", 99, muted)
        line("ROLL:    --", 120, muted)
    line(f"FPS: {fps:5.1f}   Smoothing: {smoothing:.2f}", 143, cyan, 0.48)
    line(f"C/SPACE: Calibrate  M: Mesh {'ON' if show_mesh else 'OFF'}", 161, muted, 0.40)
    cv2.putText(frame, "R: Reset  |  Q/ESC: Quit", (12, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, white, 1, cv2.LINE_AA)
