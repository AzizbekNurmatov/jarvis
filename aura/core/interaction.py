"""Hand-driven interaction events for AURA's HUD (and later OS / TARS).

Cursor + pinch state live here. Tkinter widgets, Windows input, and TARS
commands are destinations — only the HUD destination is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple
import time

from aura.core.hand_gestures import is_pinch_pose


CURSOR_MOVE = "CURSOR_MOVE"
PINCH_START = "PINCH_START"
PINCH_HOLD = "PINCH_HOLD"
PINCH_END = "PINCH_END"

DEST_HUD = "HUD"
DEST_OS = "OS"
DEST_TARS = "TARS"

INDEX_TIP = 8

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


def _is_pinch(landmarks: Sequence, holding: bool = False) -> bool:
    return is_pinch_pose(landmarks, holding=holding)


class InteractionEngine:
    """
    Turns existing Hands landmarks into cursor / pinch events.

    Does not draw, click, or send OS input. A destination layer consumes
    the events. `destinations` is reserved so OS and TARS can subscribe
    later without changing this engine.
    """

    def __init__(
        self,
        smooth: float = 0.48,
        start_frames: int = 2,
        end_frames: int = 2,
        destinations: Optional[Sequence[str]] = None,
    ):
        self.smooth = float(smooth)
        self.start_frames = max(1, int(start_frames))
        self.end_frames = max(1, int(end_frames))
        self.destinations = tuple(destinations or (DEST_HUD,))

        self._cursor: Optional[Tuple[float, float]] = None
        self._pinching = False
        self._pinch_on = 0
        self._pinch_off = 0
        self._sticky_hand = ""
        self._last_hand = ""

    def reset(self) -> None:
        self._cursor = None
        self._pinching = False
        self._pinch_on = 0
        self._pinch_off = 0
        self._sticky_hand = ""
        self._last_hand = ""

    def _emit(self, name: str, x: float, y: float, hand: str, now: float) -> Optional[InteractionEvent]:
        if DEST_HUD not in self.destinations:
            return None
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
            if landmarks is not None and _is_pinch(landmarks, holding=holding):
                return hand

        if "RIGHT" in by_label:
            return by_label["RIGHT"]
        return hands[0]

    def _smooth(self, x: float, y: float) -> Tuple[float, float]:
        if self._cursor is None:
            self._cursor = (x, y)
            return x, y
        px, py = self._cursor
        a = self.smooth
        sx = px * (1.0 - a) + x * a
        sy = py * (1.0 - a) + y * a
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

        raw_x, raw_y = _map_point(tip[0], tip[1])
        x, y = self._smooth(raw_x, raw_y)
        hand = _hand_label(chosen)
        self._last_hand = hand
        self._sticky_hand = hand

        pinching_now = bool(
            landmarks is not None
            and _is_pinch(landmarks, holding=self._pinching)
        )
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

        move = self._emit(CURSOR_MOVE, x, y, hand, now)
        if move is not None:
            events.append(move)
        if event_name is not None:
            pinch_event = self._emit(event_name, x, y, hand, now)
            if pinch_event is not None:
                events.append(pinch_event)

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
