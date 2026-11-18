"""Compact widgets anchored beside the tracked face.

Nod up / down moves a highlight. Double blink toggles the focused
widget. Nothing else — including landing on a widget — activates it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence, Tuple
import math
import time

# Face-mesh points used only to sit the stack beside the head.
_FOREHEAD = 10
_CHIN = 152
_CHEEK_A = 234
_CHEEK_B = 454

CLOCK, DATE, COUNTER = 0, 1, 2
TITLES = ("Clock", "Date", "Counter")
_MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

# BGR, matched to the HUD hex palette.
_PANEL = (48, 30, 21)          # #151e30
_PANEL_ON = (42, 78, 64)
_ACCENT = (248, 189, 56)       # #38bdf8
_GREEN = (153, 211, 52)        # #34d399
_TEXT = (255, 244, 238)        # #eef4ff
_MUTED = (186, 160, 141)       # #8da0ba
_OUTLINE = (77, 53, 38)        # #26354d


def face_stack_origin(
    landmarks: Sequence,
    width: int,
    height: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Top-left of the stack and one widget's size, in pixels.

    The stack is placed on whichever side of the face has more room
    so it sits beside the head instead of on it.
    """
    needed = max(_FOREHEAD, _CHIN, _CHEEK_A, _CHEEK_B)
    if landmarks is None or len(landmarks) <= needed:
        return None
    if width < 8 or height < 8:
        return None

    def xy(index: int) -> Tuple[float, float]:
        point = landmarks[index]
        return float(point.x) * width, float(point.y) * height

    cheek_a = xy(_CHEEK_A)
    cheek_b = xy(_CHEEK_B)
    forehead = xy(_FOREHEAD)
    chin = xy(_CHIN)

    face_left = min(cheek_a[0], cheek_b[0])
    face_right = max(cheek_a[0], cheek_b[0])
    face_top = min(forehead[1], chin[1])
    face_bottom = max(forehead[1], chin[1])
    face_w = max(face_right - face_left, 1.0)
    face_h = max(face_bottom - face_top, 1.0)

    box_h = max(32.0, min(52.0, face_h * 0.18))
    box_w = max(96.0, min(148.0, box_h * 2.75))
    gap = max(6.0, box_h * 0.16)
    stack_h = box_h * 3.0 + gap * 2.0
    margin = max(12.0, face_w * 0.14)

    if (width - face_right) >= face_left:
        x = face_right + margin
    else:
        x = face_left - margin - box_w

    y = (face_top + face_bottom) * 0.5 - stack_h * 0.5

    x = min(max(4.0, x), max(4.0, width - box_w - 4.0))
    y = min(max(4.0, y), max(4.0, height - stack_h - 4.0))
    return x, y, box_w, box_h


class HeadWidgets:
    """Focus, activation, and a smoothed anchor for the three widgets."""

    def __init__(
        self,
        select_cooldown_s: float = 0.70,
        follow_s: float = 0.09,
        clock=None,
    ):
        self.select_cooldown_s = float(select_cooldown_s)
        self.follow_s = float(follow_s) if follow_s > 1e-3 else 1e-3
        self._clock = clock or datetime.now

        self.focus = 0
        self.active = [False, False, False]
        self.count = 0
        self._select_until = 0.0
        self._anchor: Optional[Tuple[float, float, float, float]] = None
        self._anchor_t: Optional[float] = None

    def reset_anchor(self) -> None:
        self._anchor = None
        self._anchor_t = None

    def handle(self, event) -> None:
        """Apply one existing gesture event. Other events are ignored."""
        if event is None:
            return
        name = getattr(event, "name", "")
        now = float(getattr(event, "timestamp", 0.0) or 0.0)

        if name == "HEAD_NOD":
            step = -1 if getattr(event, "direction", "") == "up" else 1
            self.focus = (self.focus + step) % len(TITLES)
            return

        if name != "DOUBLE_BLINK":
            return
        if now < self._select_until:
            return
        self._select_until = now + self.select_cooldown_s
        index = self.focus
        turning_on = not self.active[index]
        self.active[index] = turning_on
        if turning_on and index == COUNTER:
            self.count += 1

    def value(self, index: int) -> str:
        """Result text for an active widget. Inactive widgets show their name."""
        if not self.active[index]:
            return TITLES[index]
        if index == CLOCK:
            return self._clock().strftime("%H:%M:%S")
        if index == DATE:
            moment = self._clock()
            return f"{_MONTHS[moment.month - 1]} {moment.day}"
        return str(self.count)

    def _smooth_anchor(
        self,
        target: Tuple[float, float, float, float],
        now: float,
    ) -> Tuple[float, float, float, float]:
        if self._anchor is None or self._anchor_t is None:
            self._anchor = target
            self._anchor_t = now
            return target
        dt = now - self._anchor_t
        self._anchor_t = now
        if dt <= 0.0:
            return self._anchor
        if dt > 0.4:
            self._anchor = target
            return target
        alpha = 1.0 - math.exp(-dt / self.follow_s)
        self._anchor = tuple(
            previous + (desired - previous) * alpha
            for previous, desired in zip(self._anchor, target)
        )
        return self._anchor

    def draw(self, frame, landmarks, now: Optional[float] = None) -> None:
        target = face_stack_origin(
            landmarks,
            frame.shape[1] if frame is not None else 0,
            frame.shape[0] if frame is not None else 0,
        )
        if frame is None or target is None:
            return
        now = time.perf_counter() if now is None else now
        x, y, box_w, box_h = self._smooth_anchor(target, now)
        gap = max(6.0, box_h * 0.16)
        for index in range(len(TITLES)):
            top = y + index * (box_h + gap)
            self._draw_one(
                frame,
                int(round(x)),
                int(round(top)),
                int(round(box_w)),
                int(round(box_h)),
                index,
            )

    def _draw_one(self, frame, x: int, y: int, w: int, h: int, index: int) -> None:
        import cv2
        import numpy as np

        focused = index == self.focus
        active = self.active[index]
        _blend_rect(frame, x, y, w, h, _PANEL_ON if active else _PANEL, 0.86)

        if focused:
            cv2.rectangle(
                frame, (x, y), (x + 3, y + h - 1), _ACCENT, -1,
            )

        border = _GREEN if active else (_ACCENT if focused else _OUTLINE)
        cv2.rectangle(
            frame,
            (x, y),
            (x + w - 1, y + h - 1),
            border,
            2 if focused or active else 1,
            cv2.LINE_AA,
        )

        if active:
            _put_centered(
                frame, TITLES[index].upper(),
                x + 6, y + 1, w - 10, int(h * 0.40),
                _GREEN, 0.36,
            )
            _put_centered(
                frame, self.value(index),
                x + 6, y + int(h * 0.36), w - 10, int(h * 0.60),
                _TEXT, 0.50,
            )
        else:
            color = _TEXT if focused else _MUTED
            _put_centered(
                frame, TITLES[index],
                x + 6, y, w - 10, h,
                color, 0.48,
            )


def _blend_rect(frame, x, y, w, h, color, alpha: float) -> None:
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    roi = frame[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def _put_centered(frame, text, x, y, w, h, color, scale: float) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    scale = float(scale)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    while tw > max(8, w - 4) and scale > 0.28:
        scale *= 0.88
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx = int(x + max(0, (w - tw) / 2))
    ty = int(y + (h + th) / 2)
    cv2.putText(
        frame, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA,
    )
