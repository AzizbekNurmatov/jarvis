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


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _blend_panel(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 color=(15, 24, 34), alpha: float = 0.78) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_axes_widget(frame: np.ndarray, pose: np.ndarray | None, enabled: bool = True) -> None:
    if not enabled:
        return

    h, w = frame.shape[:2]
    cyan = (240, 216, 74)
    white = (238, 238, 238)
    muted = (165, 176, 188)

    panel_w = 228
    panel_h = 274
    x1 = max(8, w - panel_w - 8)
    y1 = 8
    x2 = w - 8
    y2 = min(h - 8, y1 + panel_h)

    _blend_panel(frame, x1, y1, x2, y2)

    cv2.putText(frame, "HEAD AXES", (x1 + 12, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, cyan, 1, cv2.LINE_AA)

    # -----------------------------
    # YAW / PITCH XY plane
    # -----------------------------
    plane_size = 148
    plane_x = x1 + 20
    plane_y = y1 + 38
    plane_x2 = plane_x + plane_size
    plane_y2 = plane_y + plane_size
    cx = plane_x + plane_size // 2
    cy = plane_y + plane_size // 2
    margin = 12

    cv2.rectangle(frame, (plane_x, plane_y), (plane_x2, plane_y2), (80, 95, 110), 1)
    cv2.line(frame, (cx, plane_y + 4), (cx, plane_y2 - 4), muted, 1, cv2.LINE_AA)
    cv2.line(frame, (plane_x + 4, cy), (plane_x2 - 4, cy), muted, 1, cv2.LINE_AA)

    # Tick marks
    for frac in (0.25, 0.5, 0.75):
        tx = int(plane_x + frac * plane_size)
        ty = int(plane_y + frac * plane_size)
        cv2.line(frame, (tx, cy - 4), (tx, cy + 4), (95, 105, 118), 1, cv2.LINE_AA)
        cv2.line(frame, (cx - 4, ty), (cx + 4, ty), (95, 105, 118), 1, cv2.LINE_AA)

    cv2.putText(frame, "YAW", (plane_x + plane_size + 12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, white, 1, cv2.LINE_AA)
    cv2.putText(frame, "PITCH", (cx - 20, plane_y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, white, 1, cv2.LINE_AA)
    cv2.putText(frame, "L", (plane_x - 12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, muted, 1, cv2.LINE_AA)
    cv2.putText(frame, "R", (plane_x2 + 2, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, muted, 1, cv2.LINE_AA)
    cv2.putText(frame, "U", (cx - 5, plane_y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, muted, 1, cv2.LINE_AA)
    cv2.putText(frame, "D", (cx - 5, plane_y2 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, muted, 1, cv2.LINE_AA)

    if pose is not None:
        yaw, pitch, roll = [float(v) for v in pose]

        # These control sensitivity / range of movement inside the widget
        max_yaw = 35.0
        max_pitch = 25.0

        yaw_norm = _clamp(yaw / max_yaw, -1.0, 1.0)
        pitch_norm = _clamp(pitch / max_pitch, -1.0, 1.0)

        px = int(cx + yaw_norm * (plane_size / 2 - margin))
        py = int(cy - pitch_norm * (plane_size / 2 - margin))

        magnitude = min(1.0, (abs(yaw_norm) + abs(pitch_norm)) / 1.6)
        thickness = 2 + int(4 * magnitude)
        radius = 6 + int(4 * magnitude)

        # Cyan -> warmer as motion grows
        color = (40, 255 - int(120 * magnitude), 255)

        cv2.line(frame, (cx, cy), (px, py), color, thickness, cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, cyan, -1, cv2.LINE_AA)

        # -----------------------------
        # ROLL / TILT gauge
        # -----------------------------
        bar_x1 = x1 + 20
        bar_x2 = x2 - 20
        bar_y = y1 + 220
        bar_cx = (bar_x1 + bar_x2) // 2

        max_roll = 35.0
        roll_norm = _clamp(roll / max_roll, -1.0, 1.0)
        roll_x = int(bar_cx + roll_norm * ((bar_x2 - bar_x1) / 2 - 12))

        roll_mag = abs(roll_norm)
        roll_thickness = 2 + int(4 * roll_mag)
        roll_color = (60, 255 - int(120 * roll_mag), 255)

        cv2.putText(frame, "ROLL / TILT", (bar_x1, bar_y - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, white, 1, cv2.LINE_AA)
        cv2.line(frame, (bar_x1, bar_y), (bar_x2, bar_y), muted, 1, cv2.LINE_AA)
        cv2.line(frame, (bar_cx, bar_y - 8), (bar_cx, bar_y + 8), cyan, 1, cv2.LINE_AA)
        cv2.line(frame, (bar_cx, bar_y), (roll_x, bar_y), roll_color, roll_thickness, cv2.LINE_AA)
        cv2.circle(frame, (roll_x, bar_y), 7, roll_color, -1, cv2.LINE_AA)
        cv2.putText(frame, f"{roll:+.1f} deg", (bar_x1, bar_y + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, cyan, 1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (cx, cy), 4, cyan, -1, cv2.LINE_AA)

        bar_x1 = x1 + 20
        bar_x2 = x2 - 20
        bar_y = y1 + 220
        bar_cx = (bar_x1 + bar_x2) // 2

        cv2.putText(frame, "ROLL / TILT", (bar_x1, bar_y - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, white, 1, cv2.LINE_AA)
        cv2.line(frame, (bar_x1, bar_y), (bar_x2, bar_y), muted, 1, cv2.LINE_AA)
        cv2.line(frame, (bar_cx, bar_y - 8), (bar_cx, bar_y + 8), cyan, 1, cv2.LINE_AA)
        cv2.putText(frame, "--", (bar_x1, bar_y + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, muted, 1, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, pose: np.ndarray | None, fps: float,
               detected: bool, show_mesh: bool, show_axes: bool, smoothing: float) -> None:
    height, width = frame.shape[:2]
    panel_width = min(width - 16, 382)
    panel_height = 188

    _blend_panel(frame, 8, 8, panel_width, panel_height)

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
    line(f"C/SPACE: Calibrate   M: Mesh {'ON' if show_mesh else 'OFF'}", 163, muted, 0.40)
    line(f"G: Axes {'ON' if show_axes else 'OFF'}   R: Reset", 180, muted, 0.40)

    cv2.putText(frame, "Q/ESC: Quit", (12, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, white, 1, cv2.LINE_AA)