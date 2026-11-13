"""Windows desktop mouse destination for AURA interaction events.

Consumes the same InteractionFrame as the HUD lab. Gesture recognition
stays in hand_gestures / InteractionEngine; this module only maps events
to OS input so TARS can subscribe the same way later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import sys
import time

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

    Click: short pinch, mouse down/up on release.
    Drag: pinch held past drag_arm_s, then move with button down.

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

    def enable(self) -> None:
        self.enabled = True
        self.status = "AIMING"
        self.last_error = ""
        self._stop_reason = ""
        self._lost_at = None
        self._reset_gesture()

    def disable(self, reason: str = "DISABLED", notify: bool = True) -> None:
        was_on = self.enabled
        self._release()
        self._reset_gesture()
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
            if self._lost_at is None:
                self._lost_at = now
            self.status = "LOST"
            if now - self._lost_at >= self.lost_disable_s:
                self.disable(reason="TRACKING LOST")
            return

        self._lost_at = None
        x, y = self._to_screen(iframe.cursor_x, iframe.cursor_y)
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
            self.status = "PINCH"
        else:
            self.status = "AIMING"
