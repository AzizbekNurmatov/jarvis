"""Rule-based hand gesture classification from MediaPipe landmarks.

Accepts existing Hands landmarks (no second detector). Classification is
separate from stabilization and from UI / action dispatch so a future
TARS layer can consume GESTURE_STARTED / GESTURE_HELD / GESTURE_ENDED.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple
import time

import numpy as np


# Display / wire names kept stable for HUD and future TARS consumers.
OPEN_PALM = "OPEN_PALM"
FIST = "FIST"
POINTING = "POINTING"
PEACE_SIGN = "PEACE_SIGN"
PINCH = "PINCH"
UNKNOWN = "UNKNOWN"
NO_HAND = "NO_HAND"

NAMED_GESTURES = (OPEN_PALM, FIST, POINTING, PEACE_SIGN, PINCH)

GESTURE_LABELS = {
    OPEN_PALM: "OPEN PALM",
    FIST: "FIST",
    POINTING: "POINTING",
    PEACE_SIGN: "PEACE SIGN",
    PINCH: "PINCH",
    UNKNOWN: "UNKNOWN",
    NO_HAND: "NO HAND",
}

STATUS_DETECTING = "DETECTING"
STATUS_STABLE = "STABLE"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_NO_HAND = "NO HAND"

EVENT_STARTED = "GESTURE_STARTED"
EVENT_HELD = "GESTURE_HELD"
EVENT_ENDED = "GESTURE_ENDED"

# MediaPipe Hands indices.
_WRIST = 0
_THUMB_CMC, _THUMB_MCP, _THUMB_IP, _THUMB_TIP = 1, 2, 3, 4
_INDEX_MCP, _INDEX_PIP, _INDEX_DIP, _INDEX_TIP = 5, 6, 7, 8
_MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_DIP, _MIDDLE_TIP = 9, 10, 11, 12
_RING_MCP, _RING_PIP, _RING_DIP, _RING_TIP = 13, 14, 15, 16
_PINKY_MCP, _PINKY_PIP, _PINKY_DIP, _PINKY_TIP = 17, 18, 19, 20

_FINGERS = (
    (_INDEX_MCP, _INDEX_PIP, _INDEX_DIP, _INDEX_TIP),
    (_MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_DIP, _MIDDLE_TIP),
    (_RING_MCP, _RING_PIP, _RING_DIP, _RING_TIP),
    (_PINKY_MCP, _PINKY_PIP, _PINKY_DIP, _PINKY_TIP),
)

# Overlay colors match the existing hand skeleton (BGR).
_LEFT_COLOR = (210, 200, 80)
_RIGHT_COLOR = (220, 140, 210)
_LABEL_BG = (20, 18, 12)


@dataclass
class HandGestureResult:
    """Stabilized result for one hand (or the empty-frame primary)."""

    gesture: str = NO_HAND
    raw_gesture: str = NO_HAND
    label: str = GESTURE_LABELS[NO_HAND]
    hand: str = ""
    status: str = STATUS_NO_HAND
    event: Optional[str] = None
    agree_count: int = 0
    window_count: int = 0
    confirmed_at: float = 0.0

    @property
    def named(self) -> bool:
        return self.gesture in NAMED_GESTURES


@dataclass
class GestureLogEntry:
    gesture: str
    label: str
    hand: str
    timestamp: float


@dataclass
class GestureSnapshot:
    """One frame of recognition output. `hands` aligns with the input list."""

    hands: List[HandGestureResult] = field(default_factory=list)
    primary: HandGestureResult = field(default_factory=HandGestureResult)
    recent: List[GestureLogEntry] = field(default_factory=list)


def _as_points(landmarks: Sequence) -> np.ndarray:
    pts = np.zeros((21, 3), dtype=np.float64)
    count = min(21, len(landmarks))
    for i in range(count):
        lm = landmarks[i]
        pts[i, 0] = float(lm.x)
        pts[i, 1] = float(lm.y)
        pts[i, 2] = float(getattr(lm, "z", 0.0) or 0.0)
    return pts


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _dist2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at b, in degrees, using 2D to ignore noisy depth."""
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    na = float(np.linalg.norm(ba))
    nc = float(np.linalg.norm(bc))
    if na < 1e-8 or nc < 1e-8:
        return 180.0
    cosine = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _hand_scale(pts: np.ndarray) -> float:
    """Palm-relative size so thresholds survive camera distance changes."""
    wrist = pts[_WRIST]
    palm = max(
        _dist(wrist, pts[_MIDDLE_MCP]),
        _dist(pts[_INDEX_MCP], pts[_PINKY_MCP]),
        _dist(wrist, pts[_INDEX_MCP]),
        1e-6,
    )
    return palm


def _finger_extended(pts: np.ndarray, mcp: int, pip: int, tip: int, scale: float) -> bool:
    pip_angle = _angle(pts[mcp], pts[pip], pts[tip])
    span = _dist2(pts[mcp], pts[tip]) / scale
    tip_from_wrist = _dist2(pts[_WRIST], pts[tip])
    pip_from_wrist = _dist2(pts[_WRIST], pts[pip])
    stretched = tip_from_wrist > pip_from_wrist * 1.02
    if pip_angle >= 158.0 and span >= 0.38:
        return True
    if pip_angle >= 148.0 and span >= 0.48 and stretched:
        return True
    return False


def _finger_curled(pts: np.ndarray, mcp: int, pip: int, tip: int, scale: float) -> bool:
    pip_angle = _angle(pts[mcp], pts[pip], pts[tip])
    span = _dist2(pts[mcp], pts[tip]) / scale
    tip_from_wrist = _dist2(pts[_WRIST], pts[tip])
    pip_from_wrist = _dist2(pts[_WRIST], pts[pip])
    folded = tip_from_wrist < pip_from_wrist * 1.05
    if pip_angle <= 118.0:
        return True
    if span <= 0.34 and folded:
        return True
    return False


def _thumb_extended(pts: np.ndarray, scale: float) -> bool:
    ip_angle = _angle(pts[_THUMB_MCP], pts[_THUMB_IP], pts[_THUMB_TIP])
    abduct = _dist2(pts[_THUMB_TIP], pts[_MIDDLE_MCP]) / scale
    span = _dist2(pts[_THUMB_CMC], pts[_THUMB_TIP]) / scale
    away_from_index = _dist2(pts[_THUMB_TIP], pts[_INDEX_MCP]) / scale
    if abduct >= 0.52 and span >= 0.40:
        return True
    if ip_angle >= 150.0 and away_from_index >= 0.38 and span >= 0.36:
        return True
    return False


def _thumb_curled(pts: np.ndarray, scale: float) -> bool:
    abduct = _dist2(pts[_THUMB_TIP], pts[_MIDDLE_MCP]) / scale
    away_from_index = _dist2(pts[_THUMB_TIP], pts[_INDEX_MCP]) / scale
    ip_angle = _angle(pts[_THUMB_MCP], pts[_THUMB_IP], pts[_THUMB_TIP])
    if abduct <= 0.42 and away_from_index <= 0.36:
        return True
    if ip_angle <= 125.0 and abduct <= 0.55:
        return True
    return False


def classify_landmarks(landmarks: Optional[Sequence]) -> str:
    """
    Stateless pose → gesture name.

    Returns one of the five named gestures, or UNKNOWN.
    Callers should map missing hands to NO_HAND before this.
    Distances are normalized by palm size; 2D joint angles are used so
    a mirrored selfie feed does not need extra handedness logic.
    """
    if landmarks is None or len(landmarks) < 21:
        return UNKNOWN

    pts = _as_points(landmarks)
    scale = _hand_scale(pts)

    # Prefer 3D so a 2D overlap (thumb in front of the index) is not a pinch.
    pinch_dist = _dist(pts[_THUMB_TIP], pts[_INDEX_TIP]) / scale
    # A real pinch meets in front of the palm; a fist bunches near the wrist.
    pinch_from_palm = min(
        _dist(pts[_WRIST], pts[_THUMB_TIP]),
        _dist(pts[_WRIST], pts[_INDEX_TIP]),
    ) / scale

    fingers = []
    for mcp, pip, _dip, tip in _FINGERS:
        extended = _finger_extended(pts, mcp, pip, tip, scale)
        curled = False if extended else _finger_curled(pts, mcp, pip, tip, scale)
        fingers.append((extended, curled))

    index_ext, index_curl = fingers[0]
    middle_ext, middle_curl = fingers[1]
    ring_ext, ring_curl = fingers[2]
    pinky_ext, pinky_curl = fingers[3]
    four_ext = index_ext and middle_ext and ring_ext and pinky_ext
    four_curl = index_curl and middle_curl and ring_curl and pinky_curl
    thumb_ext = _thumb_extended(pts, scale)
    thumb_curl = _thumb_curled(pts, scale)

    # PINCH wins over POINTING / FIST. Distal tip contact is the discriminator
    # so a bent-index pinch is not swallowed by a closed-hand pose.
    # Fists bunch near ~1.0x palm from the wrist; pinches sit further out.
    if pinch_dist <= 0.22 and pinch_from_palm >= 1.10:
        return PINCH
    if (
        pinch_dist <= 0.30
        and pinch_from_palm >= 1.25
        and not four_ext
        and middle_curl
        and ring_curl
        and pinky_curl
    ):
        return PINCH

    if four_ext and not thumb_curl:
        return OPEN_PALM
    if four_ext and thumb_ext:
        return OPEN_PALM

    if index_ext and middle_ext and ring_curl and pinky_curl:
        return PEACE_SIGN

    if index_ext and middle_curl and ring_curl and pinky_curl:
        return POINTING

    if four_curl:
        return FIST

    return UNKNOWN


def _display_hand(label: str) -> str:
    text = (label or "").strip().title()
    if text == "Left":
        return "LEFT"
    if text == "Right":
        return "RIGHT"
    return text.upper() if text else ""


class _HandSlot:
    """Per-hand debounce. Raw classification stays independent of this."""

    def __init__(self, confirm_s: float, min_frames: int, window: int):
        self.confirm_s = confirm_s
        self.min_frames = min_frames
        self.raw_window: Deque[str] = deque(maxlen=max(3, window))
        self.candidate = NO_HAND
        self.candidate_since = 0.0
        self.candidate_frames = 0
        self.stable = NO_HAND
        self.confirmed_at = 0.0

    def reset(self) -> None:
        self.raw_window.clear()
        self.candidate = NO_HAND
        self.candidate_since = 0.0
        self.candidate_frames = 0
        self.stable = NO_HAND
        self.confirmed_at = 0.0

    def _agreement(self, target: str) -> Tuple[int, int]:
        if not self.raw_window:
            return 0, 0
        return sum(1 for g in self.raw_window if g == target), len(self.raw_window)

    def update(self, raw: str, hand: str, now: float) -> HandGestureResult:
        self.raw_window.append(raw)

        if raw == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate = raw
            self.candidate_since = now
            self.candidate_frames = 1

        confirmed = (
            self.candidate_frames >= self.min_frames
            and (now - self.candidate_since) >= self.confirm_s
        )
        previous = self.stable
        if confirmed and self.candidate != self.stable:
            self.stable = self.candidate
            self.confirmed_at = now

        event: Optional[str] = None
        if self.stable != previous:
            if previous in NAMED_GESTURES and self.stable not in NAMED_GESTURES:
                event = EVENT_ENDED
            elif self.stable in NAMED_GESTURES:
                event = EVENT_STARTED
            elif previous in NAMED_GESTURES:
                event = EVENT_ENDED
        elif self.stable in NAMED_GESTURES:
            event = EVENT_HELD

        pending = (not confirmed) or (self.candidate != self.stable)
        if pending:
            if self.candidate == NO_HAND and self.stable == NO_HAND:
                status = STATUS_NO_HAND
            else:
                status = STATUS_DETECTING
        elif self.stable == NO_HAND:
            status = STATUS_NO_HAND
        elif self.stable == UNKNOWN:
            status = STATUS_UNKNOWN
        else:
            status = STATUS_STABLE

        shown = self.stable
        agree_target = (
            self.candidate if status == STATUS_DETECTING else shown
        )
        agree, window = self._agreement(agree_target)

        return HandGestureResult(
            gesture=shown,
            raw_gesture=raw,
            label=GESTURE_LABELS.get(shown, shown),
            hand=hand,
            status=status,
            event=event,
            agree_count=agree,
            window_count=window,
            confirmed_at=self.confirmed_at,
        )


def _empty_result(event: Optional[str] = None) -> HandGestureResult:
    return HandGestureResult(
        gesture=NO_HAND,
        raw_gesture=NO_HAND,
        label=GESTURE_LABELS[NO_HAND],
        hand="",
        status=STATUS_NO_HAND,
        event=event,
    )


class HandGestureRecognizer:
    """
    Classify + stabilize hand poses from existing MediaPipe Hands output.

    `classify_landmarks` is the raw rule set. `update` holds a short
    per-hand history so the HUD only commits a gesture after it holds
    briefly, and only logs a change once per hold.
    """

    def __init__(
        self,
        confirm_s: float = 0.07,
        min_frames: int = 2,
        history_frames: int = 5,
        max_log: int = 5,
    ):
        self.confirm_s = confirm_s
        self.min_frames = min_frames
        self.history_frames = history_frames
        self.max_log = max_log
        self._slots: Dict[str, _HandSlot] = {}
        self.recent: List[GestureLogEntry] = []

    def reset(self) -> None:
        self._slots.clear()
        self.recent.clear()

    def _slot(self, key: str) -> _HandSlot:
        slot = self._slots.get(key)
        if slot is None:
            slot = _HandSlot(
                self.confirm_s, self.min_frames, self.history_frames
            )
            self._slots[key] = slot
        return slot

    def _log(self, result: HandGestureResult, now: float) -> None:
        if result.event != EVENT_STARTED:
            return
        entry = GestureLogEntry(
            gesture=result.gesture,
            label=result.label,
            hand=result.hand,
            timestamp=now,
        )
        self.recent.insert(0, entry)
        del self.recent[self.max_log:]

    def _pick_primary(
        self,
        results: List[HandGestureResult],
    ) -> HandGestureResult:
        if not results:
            return _empty_result()

        started = [r for r in results if r.event == EVENT_STARTED]
        if started:
            return started[-1]

        named = [r for r in results if r.named]
        if named:
            named.sort(key=lambda r: r.confirmed_at)
            return named[-1]

        detecting = [r for r in results if r.status == STATUS_DETECTING]
        if detecting:
            return detecting[-1]

        return results[-1]

    def update(
        self,
        hands: Optional[Iterable] = None,
        now: Optional[float] = None,
    ) -> GestureSnapshot:
        """
        Parameters
        ----------
        hands:
            Iterable of objects with `.label` ("Left"/"Right") and
            `.landmarks` (21 MediaPipe points). Empty / None → NO_HAND.
        now:
            Optional timestamp (perf counter). Tests pass this explicitly.
        """
        now = time.perf_counter() if now is None else now
        hand_list = list(hands or [])
        seen: List[str] = []
        results: List[HandGestureResult] = []
        used_labels = set()

        for hand in hand_list:
            raw_label = str(getattr(hand, "label", "") or "Right")
            display = _display_hand(raw_label)
            key = display or "RIGHT"
            if key in used_labels:
                key = f"{key}2"
            used_labels.add(key)
            seen.append(key)

            landmarks = getattr(hand, "landmarks", None)
            raw = classify_landmarks(landmarks)
            result = self._slot(key).update(raw, display, now)
            self._log(result, now)
            results.append(result)

        ended_event: Optional[str] = None
        for key in list(self._slots):
            if key in seen:
                continue
            previous = self._slots[key].stable
            if previous in NAMED_GESTURES:
                ended_event = EVENT_ENDED
            del self._slots[key]

        primary = self._pick_primary(results)
        if not results:
            primary = _empty_result(ended_event)

        return GestureSnapshot(
            hands=results,
            primary=primary,
            recent=list(self.recent),
        )


def draw_hand_gesture_overlay(frame_bgr: np.ndarray, hand, result: HandGestureResult) -> None:
    """Small `RIGHT | OPEN PALM` tag near the hand, clamped to the frame."""
    if result is None or not result.named:
        return
    landmarks = getattr(hand, "landmarks", None)
    if landmarks is None or len(landmarks) < 21:
        return

    import cv2

    height, width = frame_bgr.shape[:2]
    xs = [int(lm.x * width) for lm in landmarks]
    ys = [int(lm.y * height) for lm in landmarks]
    cx = int(sum(xs) / len(xs))
    top = min(ys)

    text = f"{result.hand} | {result.label}"
    color = _LEFT_COLOR if result.hand == "LEFT" else _RIGHT_COLOR
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    x1 = int(cx - (tw + pad * 2) / 2)
    y1 = top - th - pad * 3
    x1 = max(0, min(x1, width - tw - pad * 2 - 1))
    y1 = max(0, min(y1, height - th - pad * 2 - 1))
    x2 = x1 + tw + pad * 2
    y2 = y1 + th + pad * 2
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), _LABEL_BG, -1)
    cv2.putText(
        frame_bgr,
        text,
        (x1 + pad, y2 - pad),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
