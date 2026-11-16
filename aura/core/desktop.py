"""Windows desktop mouse destination for AURA interaction events.

Consumes the same InteractionFrame as the HUD lab. Gesture recognition
stays in hand_gestures / InteractionEngine; this module only maps events
to OS input so TARS can subscribe the same way later.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
import threading
import time
from typing import Callable, List, Optional, Tuple

from aura.core.interaction import (
    PINCH_END,
    PINCH_HOLD,
    PINCH_START,
    InteractionFrame,
)


class MouseBackend:
    """OS mouse sink. Tests inject a fake; production uses Win32."""

    def bounds(self) -> Tuple[int, int, int, int]:
        raise NotImplementedError

    def move(self, x: int, y: int) -> None:
        raise NotImplementedError

    def down(self) -> None:
        raise NotImplementedError

    def up(self) -> None:
        raise NotImplementedError

    def click(self) -> None:
        self.down()
        self.up()


class NullMouse(MouseBackend):
    def bounds(self) -> Tuple[int, int, int, int]:
        return (0, 0, 1920, 1080)

    def move(self, x: int, y: int) -> None:
        return None

    def down(self) -> None:
        return None

    def up(self) -> None:
        return None


class FakeMouse(MouseBackend):
    """In-memory backend for unit tests (never touches the real cursor)."""

    def __init__(self, origin=(0, 0), size=(1920, 1080)):
        self.origin = origin
        self.size = size
        self.pos = (origin[0], origin[1])
        self.pressed = False
        self.log: List[Tuple] = []

    def bounds(self) -> Tuple[int, int, int, int]:
        return (self.origin[0], self.origin[1], self.size[0], self.size[1])

    def move(self, x: int, y: int) -> None:
        self.pos = (int(x), int(y))
        self.log.append(("move", self.pos[0], self.pos[1]))

    def down(self) -> None:
        self.pressed = True
        self.log.append(("down",) + self.pos)

    def up(self) -> None:
        self.pressed = False
        self.log.append(("up",) + self.pos)


class Win32Mouse(MouseBackend):
    """Primary-display cursor via ctypes — no extra packages."""

    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004

    def __init__(self):
        import ctypes

        self._user32 = ctypes.windll.user32

    def bounds(self) -> Tuple[int, int, int, int]:
        # Virtual desktop so a second monitor is reachable.
        x = int(self._user32.GetSystemMetrics(76))
        y = int(self._user32.GetSystemMetrics(77))
        w = int(self._user32.GetSystemMetrics(78))
        h = int(self._user32.GetSystemMetrics(79))
        if w <= 1 or h <= 1:
            w = int(self._user32.GetSystemMetrics(0))
            h = int(self._user32.GetSystemMetrics(1))
            x, y = 0, 0
        return x, y, w, h

    def move(self, x: int, y: int) -> None:
        self._user32.SetCursorPos(int(x), int(y))

    def down(self) -> None:
        self._user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    def up(self) -> None:
        self._user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def default_mouse() -> MouseBackend:
    if sys.platform == "win32":
        return Win32Mouse()
    return NullMouse()


# Short follow so camera-rate steps become a continuous cursor without
# a visible trail. Lead is one video frame of velocity at most, and only
# while the hand is actually moving, so a stop does not coast onward.
_GLIDE_TAU_S = 0.010
_GLIDE_LEAD_S = 0.010
_GLIDE_HZ = 120.0


def glide_step(
    shown: Tuple[float, float],
    target: Tuple[float, float],
    velocity: Tuple[float, float],
    dt: float,
    tau: float = _GLIDE_TAU_S,
    lead: float = _GLIDE_LEAD_S,
) -> Tuple[float, float]:
    """Ease the drawn pointer toward the latest fingertip sample.

    `lead` seconds of the current velocity cancel the easer's lag while
    speed is steady, so the cursor stays on the hand instead of behind it.
    """
    dt = dt if dt > 0.0 else 0.0
    if dt <= 0.0:
        return shown

    aim_x = target[0] + velocity[0] * lead
    aim_y = target[1] + velocity[1] * lead
    aim_x = 0.0 if aim_x < 0.0 else 1.0 if aim_x > 1.0 else aim_x
    aim_y = 0.0 if aim_y < 0.0 else 1.0 if aim_y > 1.0 else aim_y

    if abs(aim_x - shown[0]) < 8e-4 and abs(aim_y - shown[1]) < 8e-4:
        return aim_x, aim_y

    alpha = 1.0 - math.exp(-dt / tau) if tau > 1e-6 else 1.0
    return (
        shown[0] + (aim_x - shown[0]) * alpha,
        shown[1] + (aim_y - shown[1]) * alpha,
    )


@dataclass
class DesktopDiagnostics:
    fps: float = 0.0
    latency_ms: float = 0.0
    status: str = "OFF"
    screen: str = ""
    enabled: bool = False
    pinching: bool = False
    dragging: bool = False


class DesktopController:
    """
    Maps fingertip + pinch events onto the Windows cursor.

    Click: short fist, mouse down/up on release.
    Drag: fist held past drag_arm_s, then move with button down.

    Always releases the held button on disable, lost tracking, or error.
    """

    def __init__(
        self,
        mouse: Optional[MouseBackend] = None,
        sensitivity: float = 1.0,
        drag_arm_s: float = 0.24,
        click_max_s: float = 0.35,
        lost_disable_s: float = 1.0,
        on_stop: Optional[Callable[[str], None]] = None,
    ):
        self.mouse = mouse if mouse is not None else default_mouse()
        self.sensitivity = float(sensitivity)
        self.drag_arm_s = float(drag_arm_s)
        self.click_max_s = float(click_max_s)
        self.lost_disable_s = float(lost_disable_s)
        self.on_stop = on_stop

        self.enabled = False
        self.status = "OFF"
        self.latency_ms = 0.0
        self.fps = 0.0
        self.last_error = ""

        self._button_down = False
        self._dragging = False
        self._pinch_at: Optional[float] = None
        self._lost_at: Optional[float] = None
        self._stop_reason = ""

        self._lock = threading.Lock()
        self._glide_stop = threading.Event()
        self._glide_thread: Optional[threading.Thread] = None
        self._aim: Optional[Tuple[float, float]] = None
        self._vel = (0.0, 0.0)
        self._aim_at: Optional[float] = None
        self._shown: Optional[Tuple[float, float]] = None
        self._last_px: Optional[Tuple[int, int]] = None

    def enable(self) -> None:
        self.enabled = True
        self.status = "AIMING"
        self.last_error = ""
        self._stop_reason = ""
        self._lost_at = None
        self._reset_gesture()
        self._clear_aim()
        self._start_glide()

    def disable(self, reason: str = "DISABLED", notify: bool = True) -> None:
        was_on = self.enabled
        self._stop_glide()
        self._release()
        self._reset_gesture()
        self._clear_aim()
        self.enabled = False
        self.status = "OFF"
        self._lost_at = None
        self._stop_reason = reason
        if was_on and notify and self.on_stop is not None:
            self.on_stop(reason)

    def emergency_stop(self) -> None:
        self.disable(reason="EMERGENCY STOP")

    def _reset_gesture(self) -> None:
        self._dragging = False
        self._pinch_at = None

    def _gliding(self) -> bool:
        thread = self._glide_thread
        return thread is not None and thread.is_alive()

    def _start_glide(self) -> None:
        # Tests inject a fake mouse and move it inline. Only the real
        # cursor needs a high-rate follow between camera frames.
        if not isinstance(self.mouse, Win32Mouse):
            return
        if self._gliding():
            return
        self._glide_stop.clear()
        self._glide_thread = threading.Thread(
            target=self._glide_loop,
            name="aura-pointer",
            daemon=True,
        )
        self._glide_thread.start()

    def _stop_glide(self) -> None:
        self._glide_stop.set()
        thread = self._glide_thread
        self._glide_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.2)

    def _clear_aim(self) -> None:
        with self._lock:
            self._aim = None
            self._vel = (0.0, 0.0)
            self._aim_at = None
            self._shown = None
            self._last_px = None

    def _set_aim(self, nx: float, ny: float, now: float) -> bool:
        """Store the latest fingertip. Returns True when the pointer should snap."""
        with self._lock:
            snapped = self._shown is None or self._aim_at is None
            if self._aim is not None and self._aim_at is not None:
                dt = now - self._aim_at
                if 1e-4 < dt < 0.25:
                    inst_vx = (nx - self._aim[0]) / dt
                    inst_vy = (ny - self._aim[1]) / dt
                    self._vel = (
                        self._vel[0] * 0.35 + inst_vx * 0.65,
                        self._vel[1] * 0.35 + inst_vy * 0.65,
                    )
                elif dt >= 0.25:
                    self._vel = (0.0, 0.0)
            else:
                self._vel = (0.0, 0.0)
            self._aim = (nx, ny)
            self._aim_at = now
            if snapped:
                self._shown = (nx, ny)
            return snapped

    def _glide_loop(self) -> None:
        period = 1.0 / _GLIDE_HZ
        last = time.perf_counter()
        while not self._glide_stop.wait(period):
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt <= 0.0:
                continue
            if dt > 0.05:
                dt = 0.05
            with self._lock:
                if not self.enabled or self._aim is None or self._shown is None:
                    continue
                shown = glide_step(self._shown, self._aim, self._vel, dt)
                self._shown = shown
                try:
                    sx, sy = self._to_screen(shown[0], shown[1])
                except Exception:
                    continue
                if (sx, sy) == self._last_px:
                    continue
                self._last_px = (sx, sy)
                try:
                    self.mouse.move(sx, sy)
                except Exception as exc:
                    self.last_error = str(exc) or exc.__class__.__name__

    def _release(self) -> None:
        if not self._button_down:
            self._dragging = False
            return
        try:
            self.mouse.up()
        except Exception:
            pass
        self._button_down = False
        self._dragging = False

    def _to_screen(self, nx: float, ny: float) -> Tuple[int, int]:
        ox, oy, width, height = self.mouse.bounds()
        nx = 0.5 + (float(nx) - 0.5) * self.sensitivity
        ny = 0.5 + (float(ny) - 0.5) * self.sensitivity
        nx = 0.0 if nx < 0.0 else 1.0 if nx > 1.0 else nx
        ny = 0.0 if ny < 0.0 else 1.0 if ny > 1.0 else ny
        x = ox + int(round(nx * max(width - 1, 0)))
        y = oy + int(round(ny * max(height - 1, 0)))
        return x, y

    def diagnostics(self) -> DesktopDiagnostics:
        ox, oy, width, height = self.mouse.bounds()
        return DesktopDiagnostics(
            fps=self.fps,
            latency_ms=self.latency_ms,
            status=self.status,
            screen=f"{width}x{height}",
            enabled=self.enabled,
            pinching=self._pinch_at is not None and self.enabled,
            dragging=self._dragging,
        )

    def apply(
        self,
        iframe: Optional[InteractionFrame],
        now: Optional[float] = None,
        fps: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        now = time.perf_counter() if now is None else now
        self.fps = float(fps)
        started = time.perf_counter()
        try:
            self._apply(iframe or InteractionFrame(), now)
        except Exception as exc:
            self.last_error = str(exc) or exc.__class__.__name__
            self._release()
            self.disable(reason="ERROR")
            return
        sample = (time.perf_counter() - started) * 1000.0
        if sample >= 0.0:
            if self.latency_ms:
                self.latency_ms = 0.7 * self.latency_ms + 0.3 * sample
            else:
                self.latency_ms = sample

    def _apply(self, iframe: InteractionFrame, now: float) -> None:
        if not iframe.visible:
            self._release()
            self._reset_gesture()
            self._clear_aim()
            if self._lost_at is None:
                self._lost_at = now
            self.status = "LOST"
            if now - self._lost_at >= self.lost_disable_s:
                self.disable(reason="TRACKING LOST")
            return

        self._lost_at = None
        nx = float(iframe.cursor_x)
        ny = float(iframe.cursor_y)
        snap = self._set_aim(nx, ny, now)
        # The glide thread owns the real cursor between frames. Fakes and
        # the first sample after the hand appears still move immediately.
        if snap or not self._gliding():
            x, y = self._to_screen(nx, ny)
            self._last_px = (x, y)
            self.mouse.move(x, y)

        for event in iframe.events:
            if event.name == PINCH_START:
                self._pinch_at = event.timestamp or now
                self._dragging = False
            elif event.name == PINCH_HOLD:
                if (
                    not self._dragging
                    and self._pinch_at is not None
                    and (now - self._pinch_at) >= self.drag_arm_s
                ):
                    self.mouse.down()
                    self._button_down = True
                    self._dragging = True
            elif event.name == PINCH_END:
                held = (now - self._pinch_at) if self._pinch_at else 0.0
                if self._button_down or self._dragging:
                    self._release()
                elif self._pinch_at is not None and held <= self.click_max_s:
                    self.mouse.click()
                self._reset_gesture()

        if self._dragging:
            self.status = "DRAG"
        elif iframe.pinching:
            self.status = "FIST"
        else:
            self.status = "AIMING"
