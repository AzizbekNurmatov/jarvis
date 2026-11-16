"""Hand-driven interaction events for AURA's HUD (and later OS / TARS).

Cursor + fist-grab state live here. Tkinter widgets, Windows input, and TARS
commands are destinations — only the HUD destination is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Iterable, List, Optional, Sequence, Tuple

from aura.core.hand_gestures import is_fist_pose


CURSOR_MOVE = "CURSOR_MOVE"
PINCH_START = "PINCH_START"
PINCH_HOLD = "PINCH_HOLD"
PINCH_END = "PINCH_END"

DEST_HUD = "HUD"
DEST_OS = "OS"
DEST_TARS = "TARS"

INDEX_TIP = 8
_PALM = (0, 5, 9, 13, 17)

# Inset so the user does not have to reach the camera corners.
_EDGE = 0.10


@dataclass
class InteractionEvent:
    name: str
    x: float
    y: float
    hand: str = ""
    timestamp: float = 0.0
    destination: str = DEST_HUD


@dataclass
class InteractionFrame:
    cursor_x: float = 0.5
    cursor_y: float = 0.5
    visible: bool = False
    pinching: bool = False
    hand: str = ""
    events: List[InteractionEvent] = field(default_factory=list)


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _lowpass_alpha(cutoff_hz: float, dt: float) -> float:
    """Smoothing weight for a first-order low-pass at this sample interval."""
    cutoff_hz = cutoff_hz if cutoff_hz > 1e-3 else 1e-3
    dt = dt if dt > 1e-4 else 1e-4
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class _OneEuro:
    """Speed-adaptive low-pass (the 1€ filter).

    A fixed blend either trails a fast hand or shakes while the hand is
    still. This one raises its cutoff as the fingertip speeds up, so a
    deliberate move stays on the finger and a resting hand stays put.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._y: Optional[float] = None
        self._rate = 0.0
        self._t: Optional[float] = None

    def reset(self) -> None:
        self._y = None
        self._rate = 0.0
        self._t = None

    def __call__(self, value: float, timestamp: float) -> float:
        if self._y is None or self._t is None:
            self._y = float(value)
            self._rate = 0.0
            self._t = float(timestamp)
            return self._y

        dt = float(timestamp) - self._t
        if dt <= 1e-4:
            return self._y
        # A tracking gap should snap, not smear the cursor across the screen.
        if dt > 0.4:
            self._y = float(value)
            self._rate = 0.0
            self._t = float(timestamp)
            return self._y

        rate = (float(value) - self._y) / dt
        self._rate += _lowpass_alpha(self.d_cutoff, dt) * (rate - self._rate)
        cutoff = self.min_cutoff + self.beta * abs(self._rate)
        self._y += _lowpass_alpha(cutoff, dt) * (float(value) - self._y)
        self._t = float(timestamp)
        return self._y


def _map_point(x: float, y: float) -> Tuple[float, float]:
    span = 1.0 - 2.0 * _EDGE
    if span <= 1e-6:
        return _clamp01(x), _clamp01(y)
    return (
        _clamp01((x - _EDGE) / span),
        _clamp01((y - _EDGE) / span),
    )


def _tip_xy(landmarks: Sequence) -> Optional[Tuple[float, float]]:
    if landmarks is None or len(landmarks) <= INDEX_TIP:
        return None
    tip = landmarks[INDEX_TIP]
    return float(tip.x), float(tip.y)


def _hand_label(hand) -> str:
    raw = str(getattr(hand, "label", "") or "").strip().title()
    if raw == "Left":
        return "LEFT"
    if raw == "Right":
        return "RIGHT"
    return raw.upper() if raw else "RIGHT"


def _palm_xy(landmarks: Sequence) -> Optional[Tuple[float, float]]:
    if landmarks is None or len(landmarks) < 18:
        return None
    xs = 0.0
    ys = 0.0
    for index in _PALM:
        pt = landmarks[index]
        xs += float(pt.x)
        ys += float(pt.y)
    n = float(len(_PALM))
    return xs / n, ys / n


def _is_grab(landmarks: Sequence, holding: bool = False) -> bool:
    return is_fist_pose(landmarks, holding=holding)


class InteractionEngine:
    """
    Turns existing Hands landmarks into cursor / fist-grab events.

    Does not draw, click, or send OS input. Callers fan the same
    InteractionFrame out to HUD, Desktop Control, and later TARS.
    """

    def __init__(
        self,
        smooth: float = 0.48,
        start_frames: int = 2,
        end_frames: int = 2,
        destinations: Optional[Sequence[str]] = None,
        min_cutoff: float = 1.05,
        beta: float = 24.0,
        d_cutoff: float = 1.0,
    ):
        # smooth >= 1 keeps the raw fingertip (tests). Anything lower uses
        # the adaptive filter; the old fixed blend is what felt late and shaky.
        self.smooth = float(smooth)
        self._passthrough = self.smooth >= 0.999
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.start_frames = max(1, int(start_frames))
        self.end_frames = max(1, int(end_frames))
        self.destinations = tuple(destinations or (DEST_HUD,))

        self._cursor: Optional[Tuple[float, float]] = None
        self._pinching = False
        self._pinch_on = 0
        self._pinch_off = 0
        self._sticky_hand = ""
        self._last_hand = ""
        self._fist_offset: Optional[Tuple[float, float]] = None
        self._fx = _OneEuro(self.min_cutoff, self.beta, self.d_cutoff)
        self._fy = _OneEuro(self.min_cutoff, self.beta, self.d_cutoff)

    def reset(self) -> None:
        self._cursor = None
        self._pinching = False
        self._pinch_on = 0
        self._pinch_off = 0
        self._sticky_hand = ""
        self._last_hand = ""
        self._fist_offset = None
        self._fx.reset()
        self._fy.reset()

    def _emit(self, name: str, x: float, y: float, hand: str, now: float) -> InteractionEvent:
        return InteractionEvent(
            name=name,
            x=x,
            y=y,
            hand=hand,
            timestamp=now,
            destination=DEST_HUD,
        )

    def _pick(self, hands: List) -> Optional[object]:
        if not hands:
            return None

        by_label = {_hand_label(h): h for h in hands}

        if self._sticky_hand and self._sticky_hand in by_label:
            return by_label[self._sticky_hand]

        for hand in hands:
            landmarks = getattr(hand, "landmarks", None)
            holding = (
                self._pinching
                and _hand_label(hand) == self._sticky_hand
            )
            if landmarks is not None and _is_grab(landmarks, holding=holding):
                return hand

        if "RIGHT" in by_label:
            return by_label["RIGHT"]
        return hands[0]

    def _smooth(self, x: float, y: float, now: float) -> Tuple[float, float]:
        if self._passthrough:
            self._cursor = (x, y)
            return x, y
        sx = self._fx(x, now)
        sy = self._fy(y, now)
        self._cursor = (sx, sy)
        return sx, sy

    def update(
        self,
        hands: Optional[Iterable] = None,
        now: Optional[float] = None,
    ) -> InteractionFrame:
        now = time.perf_counter() if now is None else now
        hand_list = list(hands or [])
        chosen = self._pick(hand_list)
        events: List[InteractionEvent] = []

        if chosen is None:
            ended = None
            if self._pinching:
                x, y = self._cursor or (0.5, 0.5)
                ended = self._emit(
                    PINCH_END, x, y, self._last_hand, now
                )
            self.reset()
            frame = InteractionFrame(events=[ended] if ended else [])
            return frame

        landmarks = getattr(chosen, "landmarks", None)
        tip = _tip_xy(landmarks) if landmarks is not None else None
        if tip is None:
            if self._pinching:
                x, y = self._cursor or (0.5, 0.5)
                ended = self._emit(
                    PINCH_END, x, y, self._last_hand, now
                )
                self.reset()
                return InteractionFrame(events=[ended] if ended else [])
            self.reset()
            return InteractionFrame()

        mapped_tip = _map_point(tip[0], tip[1])
        palm = _palm_xy(landmarks) if landmarks is not None else None
        mapped_palm = _map_point(palm[0], palm[1]) if palm is not None else mapped_tip

        grab_now = bool(
            landmarks is not None
            and _is_grab(landmarks, holding=self._pinching)
        )
        # Aim with the index tip; while the fist is closed, follow the palm
        # so the cursor does not jump into the knuckles.
        if grab_now or self._pinching:
            if self._fist_offset is None:
                cx, cy = self._cursor if self._cursor is not None else mapped_tip
                self._fist_offset = (cx - mapped_palm[0], cy - mapped_palm[1])
            raw_x = _clamp01(mapped_palm[0] + self._fist_offset[0])
            raw_y = _clamp01(mapped_palm[1] + self._fist_offset[1])
        else:
            self._fist_offset = None
            raw_x, raw_y = mapped_tip

        x, y = self._smooth(raw_x, raw_y, now)
        hand = _hand_label(chosen)
        self._last_hand = hand
        self._sticky_hand = hand

        pinching_now = grab_now
        event_name = None

        if pinching_now:
            self._pinch_off = 0
            self._pinch_on += 1
            if not self._pinching and self._pinch_on >= self.start_frames:
                self._pinching = True
                event_name = PINCH_START
            elif self._pinching:
                event_name = PINCH_HOLD
        else:
            self._pinch_on = 0
            if self._pinching:
                self._pinch_off += 1
                if self._pinch_off >= self.end_frames:
                    self._pinching = False
                    self._pinch_off = 0
                    event_name = PINCH_END
            else:
                self._pinch_off = 0

        events.append(self._emit(CURSOR_MOVE, x, y, hand, now))
        if event_name is not None:
            events.append(self._emit(event_name, x, y, hand, now))

        return InteractionFrame(
            cursor_x=x,
            cursor_y=y,
            visible=True,
            pinching=self._pinching,
            hand=hand,
            events=events,
        )


def draw_index_pointer(frame_bgr, iframe: InteractionFrame) -> None:
    """Small fingertip marker on the camera while Interaction Lab is on."""
    if frame_bgr is None or not iframe.visible:
        return
    import cv2

    height, width = frame_bgr.shape[:2]
    # Pointer is drawn in camera space; invert the lab inset so it sits
    # on the actual index tip rather than the mapped lab coordinate.
    span = 1.0 - 2.0 * _EDGE
    nx = iframe.cursor_x * span + _EDGE
    ny = iframe.cursor_y * span + _EDGE
    px = int(nx * width)
    py = int(ny * height)
    color = (80, 220, 140) if iframe.pinching else (248, 189, 56)
    cv2.circle(frame_bgr, (px, py), 11, color, 2, cv2.LINE_AA)
    cv2.circle(frame_bgr, (px, py), 3, color, -1, cv2.LINE_AA)
