"""Reusable one-shot gesture recognition from blink + head pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

from aura.core.pose import wrap_angle


# Display labels kept stable for HUD / future TARS wiring.
GESTURE_LABELS = {
    "LEFT_WINK": "LEFT WINK",
    "RIGHT_WINK": "RIGHT WINK",
    "BOTH_BLINK": "BOTH BLINK",
    "DOUBLE_BLINK": "DOUBLE BLINK",
    "HEAD_NOD": "HEAD NOD",
    "HEAD_SHAKE": "HEAD SHAKE",
    "HEAD_TURN_LEFT": "HEAD TURN LEFT",
    "HEAD_TURN_RIGHT": "HEAD TURN RIGHT",
}


@dataclass
class GestureEvent:
    name: str
    label: str
    timestamp: float


class GestureEngine:
    """
    Consumes blink events and relative head rotation.

    Emits at most one discrete gesture event per recognition.
    Thresholds are instance attributes so they are easy to tune.
    """

    def __init__(
        self,
        double_blink_window_s=0.55,
        both_defer_s=0.32,
        nod_down_deg=12.0,
        nod_return_deg=6.0,
        nod_window_s=0.85,
        shake_deg=12.0,
        shake_return_deg=6.0,
        shake_window_s=1.1,
        turn_deg=18.0,
        turn_hold_s=0.35,
        turn_return_deg=10.0,
        cooldown_s=0.55,
        head_cooldown_s=0.75,
    ):
        self.double_blink_window_s = double_blink_window_s
        self.both_defer_s = both_defer_s

        self.nod_down_deg = nod_down_deg
        self.nod_return_deg = nod_return_deg
        self.nod_window_s = nod_window_s

        self.shake_deg = shake_deg
        self.shake_return_deg = shake_return_deg
        self.shake_window_s = shake_window_s

        self.turn_deg = turn_deg
        self.turn_hold_s = turn_hold_s
        self.turn_return_deg = turn_return_deg

        self.cooldown_s = cooldown_s
        self.head_cooldown_s = head_cooldown_s

        self.latest: Optional[GestureEvent] = None
        self.status = "Ready"
        self.detecting: Optional[str] = None

        self._cooldown_until = 0.0
        self._head_cooldown_until = 0.0

        self._pending_both_at = None
        self._last_both_at = None

        self._nod_started_at = None
        self._nod_peak = 0.0

        self._shake_started_at = None
        self._shake_seen_left = False
        self._shake_seen_right = False

        self._turn_left_since = None
        self._turn_right_since = None
        self._turn_left_fired = False
        self._turn_right_fired = False

    def reset(self):
        self.latest = None
        self.status = "Ready"
        self.detecting = None
        self._cooldown_until = 0.0
        self._head_cooldown_until = 0.0
        self._pending_both_at = None
        self._last_both_at = None
        self._reset_head_partial()

    def _reset_head_partial(self):
        self._nod_started_at = None
        self._nod_peak = 0.0
        self._shake_started_at = None
        self._shake_seen_left = False
        self._shake_seen_right = False
        self._turn_left_since = None
        self._turn_right_since = None
        self._turn_left_fired = False
        self._turn_right_fired = False

    def _emit(self, name: str, now: float, head: bool = False) -> GestureEvent:
        event = GestureEvent(
            name=name,
            label=GESTURE_LABELS[name],
            timestamp=now,
        )
        self.latest = event
        self.status = "Cooldown"
        self.detecting = None
        self._cooldown_until = now + self.cooldown_s
        if head:
            self._head_cooldown_until = now + self.head_cooldown_s
            self._reset_head_partial()
        return event

    def _update_status(self, now: float):
        if now < self._cooldown_until or now < self._head_cooldown_until:
            self.status = "Cooldown"
        elif self.detecting or self._pending_both_at is not None:
            self.status = "Detecting"
        else:
            self.status = "Ready"

    def _handle_blink_event(self, blink_label: str, now: float) -> Optional[GestureEvent]:
        if blink_label == "Left Eye":
            self._pending_both_at = None
            return self._emit("LEFT_WINK", now)

        if blink_label == "Right Eye":
            self._pending_both_at = None
            return self._emit("RIGHT_WINK", now)

        if blink_label == "Both Eyes":
            if (
                self._last_both_at is not None
                and now - self._last_both_at <= self.double_blink_window_s
            ):
                self._pending_both_at = None
                self._last_both_at = None
                return self._emit("DOUBLE_BLINK", now)

            # Defer BOTH so a quick second blink can become DOUBLE.
            self._pending_both_at = now
            self._last_both_at = now
            self.detecting = "DOUBLE BLINK?"
            return None

        return None

    def _flush_pending_both(self, now: float) -> Optional[GestureEvent]:
        if self._pending_both_at is None:
            return None
        if now - self._pending_both_at < self.both_defer_s:
            self.detecting = "DOUBLE BLINK?"
            return None
        self._pending_both_at = None
        return self._emit("BOTH_BLINK", now)

    def _update_head(self, yaw: float, pitch: float, now: float) -> Optional[GestureEvent]:
        if now < self._head_cooldown_until or now < self._cooldown_until:
            return None

        yaw = wrap_angle(yaw)
        pitch = wrap_angle(pitch)

        # --- Nod: pitch away from neutral then return ---
        # Accept either sign; solvePnP pitch direction varies by setup.
        if abs(pitch) >= self.nod_down_deg:
            if self._nod_started_at is None:
                self._nod_started_at = now
                self._nod_peak = pitch
                self.detecting = "HEAD NOD"
            elif abs(pitch) > abs(self._nod_peak):
                self._nod_peak = pitch
        elif self._nod_started_at is not None:
            if (
                abs(pitch) <= self.nod_return_deg
                and abs(self._nod_peak) >= self.nod_down_deg
                and now - self._nod_started_at <= self.nod_window_s
            ):
                return self._emit("HEAD_NOD", now, head=True)
            if now - self._nod_started_at > self.nod_window_s:
                self._nod_started_at = None
                self._nod_peak = 0.0

        # --- Shake: both yaw directions, then return ---
        if abs(yaw) >= self.shake_deg:
            if self._shake_started_at is None:
                self._shake_started_at = now
                self.detecting = self.detecting or "HEAD SHAKE"
            if yaw <= -self.shake_deg:
                self._shake_seen_left = True
            if yaw >= self.shake_deg:
                self._shake_seen_right = True
        elif self._shake_started_at is not None:
            if (
                self._shake_seen_left
                and self._shake_seen_right
                and abs(yaw) <= self.shake_return_deg
                and now - self._shake_started_at <= self.shake_window_s
            ):
                return self._emit("HEAD_SHAKE", now, head=True)
            if now - self._shake_started_at > self.shake_window_s:
                self._shake_started_at = None
                self._shake_seen_left = False
                self._shake_seen_right = False

        # --- Held turns ---
        if yaw <= -self.turn_deg:
            if self._turn_left_since is None:
                self._turn_left_since = now
                self.detecting = "HEAD TURN LEFT"
            elif (
                not self._turn_left_fired
                and now - self._turn_left_since >= self.turn_hold_s
            ):
                self._turn_left_fired = True
                return self._emit("HEAD_TURN_LEFT", now, head=True)
        else:
            if abs(yaw) <= self.turn_return_deg:
                self._turn_left_since = None
                self._turn_left_fired = False

        if yaw >= self.turn_deg:
            if self._turn_right_since is None:
                self._turn_right_since = now
                self.detecting = "HEAD TURN RIGHT"
            elif (
                not self._turn_right_fired
                and now - self._turn_right_since >= self.turn_hold_s
            ):
                self._turn_right_fired = True
                return self._emit("HEAD_TURN_RIGHT", now, head=True)
        else:
            if abs(yaw) <= self.turn_return_deg:
                self._turn_right_since = None
                self._turn_right_fired = False

        return None

    def update(self, blink_result, rotation, now=None) -> Optional[GestureEvent]:
        """
        Parameters
        ----------
        blink_result:
            BlinkResult (or None when face is lost).
        rotation:
            Iterable [yaw, pitch, roll] in degrees relative to neutral.
        """
        now = time.perf_counter() if now is None else now
        event = None

        if blink_result is not None and blink_result.detected:
            if now >= self._cooldown_until:
                event = self._handle_blink_event(blink_result.latest, now)

        if event is None:
            event = self._flush_pending_both(now)

        if event is None and rotation is not None:
            yaw = float(rotation[0])
            pitch = float(rotation[1])
            event = self._update_head(yaw, pitch, now)

        if event is None and self.detecting and self.status != "Cooldown":
            # Clear stale detecting label when nothing is in progress.
            if (
                self._pending_both_at is None
                and self._nod_started_at is None
                and self._shake_started_at is None
                and self._turn_left_since is None
                and self._turn_right_since is None
            ):
                self.detecting = None

        self._update_status(now)
        return event
