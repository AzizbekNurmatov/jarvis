
"""AURA v0.2 - Wide spatial tracking dashboard."""

from __future__ import annotations

import cv2
import numpy as np


# Dashboard dimensions
WIDTH = 1240
HEIGHT = 720

# OpenCV uses BGR colors
BG = (22, 17, 12)
PANEL = (39, 32, 23)
PANEL_INNER = (48, 40, 29)
BORDER = (75, 65, 49)
CYAN = (235, 221, 91)
WHITE = (240, 240, 240)
MUTED = (174, 167, 155)
GREEN = (125, 225, 130)
RED = (110, 110, 250)
ORANGE = (80, 175, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


CONTOURS = [
    [10, 338, 297, 332, 284, 251, 389, 356,
     454, 323, 361, 288, 397, 365, 379, 378,
     400, 377, 152, 148, 176, 149, 150, 136,
     172, 58, 132, 93, 234, 127, 162, 21,
     54, 103, 67, 109, 10],

    [33, 160, 158, 133, 153, 144, 33],
    [362, 385, 387, 263, 373, 380, 362],
    [61, 40, 37, 0, 267, 270, 291,
     321, 314, 17, 84, 91, 61],
    [168, 6, 197, 195, 5, 4, 1, 2, 98],
]


def text(
    img,
    value,
    x,
    y,
    size=0.52,
    color=WHITE,
    thickness=1,
):
    cv2.putText(
        img,
        str(value),
        (int(x), int(y)),
        FONT,
        size,
        color,
        thickness,
        cv2.LINE_AA,
    )


def rect(img, x1, y1, x2, y2, color=PANEL):
    cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        -1,
    )


def outline(img, x1, y1, x2, y2, color=BORDER):
    cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        1,
        cv2.LINE_AA,
    )


def clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, float(value)))


def draw_mesh(frame, landmarks):
    h, w = frame.shape[:2]

    for contour in CONTOURS:
        points = []

        for idx in contour:
            if idx < len(landmarks):
                lm = landmarks[idx]
                points.append(
                    (int(lm.x * w), int(lm.y * h))
                )

        if len(points) > 1:
            cv2.polylines(
                frame,
                [np.array(points, dtype=np.int32)],
                False,
                CYAN,
                1,
                cv2.LINE_AA,
            )

    for idx in range(0, min(len(landmarks), 468), 5):
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)

        if 0 <= x < w and 0 <= y < h:
            cv2.circle(
                frame,
                (x, y),
                1,
                ORANGE,
                -1,
                cv2.LINE_AA,
            )


def draw_card(canvas, x1, y1, x2, y2, title):
    rect(canvas, x1, y1, x2, y2, PANEL)
    outline(canvas, x1, y1, x2, y2)

    text(
        canvas, title, x1 + 16, y1 + 27,
        0.53, CYAN, 1
    )


def draw_position_map(canvas, position):
    """Large X/Y positional map."""

    x1, y1 = 850, 78
    x2, y2 = 1216, 366

    draw_card(
        canvas, x1, y1, x2, y2,
        "POSITION / X-Y"
    )

    # Plot area
    px1, py1 = 915, 127
    px2, py2 = 1148, 333

    rect(
        canvas, px1, py1, px2, py2,
        PANEL_INNER
    )
    outline(canvas, px1, py1, px2, py2)

    cx = (px1 + px2) // 2
    cy = (py1 + py2) // 2

    # Grid lines
    for fraction in (0.25, 0.5, 0.75):
        gx = int(px1 + (px2 - px1) * fraction)
        gy = int(py1 + (py2 - py1) * fraction)

        cv2.line(
            canvas,
            (gx, py1),
            (gx, py2),
            BORDER,
            1,
            cv2.LINE_AA,
        )

        cv2.line(
            canvas,
            (px1, gy),
            (px2, gy),
            BORDER,
            1,
            cv2.LINE_AA,
        )

    text(canvas, "UP", cx - 12, py1 - 9, 0.38, MUTED)
    text(canvas, "DOWN", cx - 20, py2 + 19, 0.38, MUTED)
    text(canvas, "LEFT", px1 - 47, cy + 5, 0.38, MUTED)
    text(canvas, "RIGHT", px2 + 5, cy + 5, 0.38, MUTED)

    cv2.circle(canvas, (cx, cy), 4, MUTED, -1)

    if position is None:
        text(
            canvas,
            "NO POSITION",
            px1 + 54,
            py1 + 105,
            0.5,
            RED,
        )
        return

    x, y, z = position

    # Sensitivity: +/- 20% of image width/height
    x_norm = clamp(x / 20.0)
    y_norm = clamp(y / 20.0)

    radius_x = (px2 - px1) // 2 - 12
    radius_y = (py2 - py1) // 2 - 12

    dot_x = int(cx + x_norm * radius_x)
    dot_y = int(cy - y_norm * radius_y)

    strength = min(
        1.0,
        float(np.hypot(x_norm, y_norm))
    )

    thickness = 2 + int(strength * 5)
    dot_radius = 7 + int(strength * 5)

    color = CYAN if strength < 0.7 else ORANGE

    cv2.line(
        canvas,
        (cx, cy),
        (dot_x, dot_y),
        color,
        thickness,
        cv2.LINE_AA,
    )

    cv2.circle(
        canvas,
        (dot_x, dot_y),
        dot_radius + 5,
        color,
        1,
        cv2.LINE_AA,
    )

    cv2.circle(
        canvas,
        (dot_x, dot_y),
        dot_radius,
        color,
        -1,
        cv2.LINE_AA,
    )

    text(
        canvas, f"X {x:+.1f}%",
        x1 + 15, y2 - 12, 0.46, WHITE
    )
    text(
        canvas, f"Y {y:+.1f}%",
        x1 + 160, y2 - 12, 0.46, WHITE
    )


def draw_depth_gauge(canvas, position):
    """Relative forward/backward movement gauge."""

    x1, y1 = 850, 380
    x2, y2 = 1216, 478

    draw_card(
        canvas, x1, y1, x2, y2,
        "DEPTH / Z"
    )

    center = 1033
    bar_start = 880
    bar_end = 1187
    bar_y = 437

    cv2.line(
        canvas,
        (bar_start, bar_y),
        (bar_end, bar_y),
        BORDER,
        8,
        cv2.LINE_AA,
    )

    cv2.line(
        canvas,
        (center, bar_y - 15),
        (center, bar_y + 15),
        WHITE,
        1,
        cv2.LINE_AA,
    )

    if position is None:
        text(
            canvas, "NO DEPTH DATA",
            884, 420, 0.45, MUTED
        )
        return

    z = float(position[2])

    # +/- 40% of calibrated apparent face size
    normalized = clamp(z / 40.0)

    target = int(
        center + normalized * 140
    )

    color = CYAN if z >= 0 else ORANGE

    cv2.line(
        canvas,
        (center, bar_y),
        (target, bar_y),
        color,
        7,
        cv2.LINE_AA,
    )

    cv2.circle(
        canvas,
        (target, bar_y),
        9,
        color,
        -1,
        cv2.LINE_AA,
    )

    direction = (
        "CLOSER" if z > 2
        else "FARTHER" if z < -2
        else "NEUTRAL"
    )

    text(
        canvas,
        f"{z:+.1f}%  {direction}",
        1030, 413, 0.48, WHITE
    )

    text(
        canvas, "FAR",
        880, 468, 0.35, MUTED
    )
    text(
        canvas, "NEAR",
        1152, 468, 0.35, MUTED
    )


def draw_rotation_gauges(canvas, rotation):
    """Yaw, pitch, roll bars."""

    x1, y1 = 850, 491
    x2, y2 = 1216, 664

    draw_card(
        canvas, x1, y1, x2, y2,
        "ORIENTATION / ROTATION"
    )

    labels = ("YAW", "PITCH", "ROLL")
    max_angles = (40.0, 30.0, 40.0)
    ys = (545, 594, 643)

    for index, (label, limit, y) in enumerate(
        zip(labels, max_angles, ys)
    ):
        value = (
            float(rotation[index])
            if rotation is not None
            else 0.0
        )

        text(
            canvas,
            label,
            868,
            y - 7,
            0.46,
            MUTED,
        )

        bar_left = 957
        bar_right = 1160
        bar_center = (bar_left + bar_right) // 2

        cv2.line(
            canvas,
            (bar_left, y),
            (bar_right, y),
            BORDER,
            5,
            cv2.LINE_AA,
        )

        cv2.line(
            canvas,
            (bar_center, y - 9),
            (bar_center, y + 9),
            WHITE,
            1,
            cv2.LINE_AA,
        )

        if rotation is not None:
            normalized = clamp(value / limit)

            target = int(
                bar_center + normalized * 94
            )

            cv2.line(
                canvas,
                (bar_center, y),
                (target, y),
                CYAN,
                5,
                cv2.LINE_AA,
            )

            cv2.circle(
                canvas,
                (target, y),
                6,
                CYAN,
                -1,
                cv2.LINE_AA,
            )

        display = (
            f"{value:+.1f}"
            if rotation is not None
            else "--"
        )

        text(
            canvas,
            display,
            1168,
            y + 5,
            0.42,
            WHITE,
        )


def draw_dashboard(
    frame,
    rotation,
    position,
    fps,
    detected,
    show_mesh,
    show_gauges,
    smoothing,
):
    canvas = np.full(
        (HEIGHT, WIDTH, 3),
        BG,
        dtype=np.uint8,
    )

    # Header
    text(
        canvas,
        "AURA / SPATIAL TRACKING",
        25, 39, 0.82, CYAN, 2
    )

    status = (
        "TRACKING ACTIVE"
        if detected
        else "SEARCHING FOR FACE"
    )

    status_color = GREEN if detected else RED

    cv2.circle(
        canvas,
        (1020, 32),
        6,
        status_color,
        -1,
        cv2.LINE_AA,
    )

    text(
        canvas,
        status,
        1035, 38,
        0.48,
        status_color,
    )

    # Camera panel
    draw_card(
        canvas,
        20, 78, 831, 664,
        "LIVE CAMERA / FACE TRACKING"
    )

    # Fit webcam without stretching
    camera_left = 34
    camera_top = 120
    camera_width = 782
    camera_height = 523

    fh, fw = frame.shape[:2]

    scale = min(
        camera_width / fw,
        camera_height / fh
    )

    new_width = max(1, int(fw * scale))
    new_height = max(1, int(fh * scale))

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    offset_x = camera_left + (
        camera_width - new_width
    ) // 2

    offset_y = camera_top + (
        camera_height - new_height
    ) // 2

    canvas[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width
    ] = resized

    # Camera footer label
    text(
        canvas,
        f"CAM 01  |  {fw}x{fh}",
        35, 657, 0.43, MUTED
    )

    text(
        canvas,
        f"{fps:.1f} FPS",
        715, 657, 0.48, CYAN
    )

    if show_gauges:
        draw_position_map(canvas, position)
        draw_depth_gauge(canvas, position)
        draw_rotation_gauges(canvas, rotation)

    else:
        draw_card(
            canvas,
            850, 78, 1216, 664,
            "GAUGES HIDDEN"
        )

        text(
            canvas,
            "Press G to show tracking gauges",
            875, 160, 0.49, MUTED
        )

    # Bottom controls
    cv2.line(
        canvas,
        (20, 683),
        (1216, 683),
        BORDER,
        1,
        cv2.LINE_AA,
    )

    text(
        canvas,
        "C / SPACE  CALIBRATE",
        25, 705, 0.43, WHITE
    )

    text(
        canvas,
        f"M  MESH {'ON' if show_mesh else 'OFF'}",
        265, 705, 0.43, MUTED
    )

    text(
        canvas,
        f"G  GAUGES {'ON' if show_gauges else 'OFF'}",
        430, 705, 0.43, MUTED
    )

    text(
        canvas,
        "R  RESET",
        675, 705, 0.43, MUTED
    )

    text(
        canvas,
        "Q / ESC  QUIT",
        810, 705, 0.43, MUTED
    )

    text(
        canvas,
        f"EMA {smoothing:.2f}",
        1100, 705, 0.43, CYAN
    )

    return canvas
