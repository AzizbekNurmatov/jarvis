"""Desktop Control click / drag / safety tests (no real mouse)."""

from __future__ import annotations

import unittest

from aura.core.desktop import DesktopController, FakeMouse
from aura.core.interaction import (
    CURSOR_MOVE,
    PINCH_END,
    PINCH_HOLD,
    PINCH_START,
    InteractionEvent,
    InteractionFrame,
)


def _event(name, x=0.5, y=0.5, now=1.0):
    return InteractionEvent(name=name, x=x, y=y, timestamp=now)


def _frame(x=0.5, y=0.5, visible=True, pinching=False, events=None, hand="RIGHT"):
    return InteractionFrame(
        cursor_x=x,
        cursor_y=y,
        visible=visible,
        pinching=pinching,
        hand=hand,
        events=list(events or []),
    )


class DesktopControllerTests(unittest.TestCase):
    def setUp(self):
        self.mouse = FakeMouse(size=(1920, 1080))
        self.stopped = []
        self.ctl = DesktopController(
            mouse=self.mouse,
            sensitivity=1.0,
            drag_arm_s=0.24,
            click_max_s=0.35,
            lost_disable_s=0.50,
            on_stop=lambda reason: self.stopped.append(reason),
        )

    def _kinds(self):
        return [row[0] for row in self.mouse.log]

    def test_disabled_by_default_is_noop(self):
        self.ctl.apply(_frame(events=[_event(CURSOR_MOVE)]), now=1.0)
        self.assertEqual(self.mouse.log, [])
        self.assertFalse(self.ctl.enabled)

    def test_maps_center_and_sensitivity(self):
        self.ctl.enable()
        self.ctl.apply(_frame(0.5, 0.5, events=[_event(CURSOR_MOVE)]), now=1.0)
        self.assertEqual(self.mouse.pos, (960, 540))

        self.mouse.log.clear()
        self.ctl.sensitivity = 1.4
        self.ctl.apply(_frame(0.75, 0.5, events=[_event(CURSOR_MOVE)]), now=1.1)
        far = self.mouse.pos[0]
        self.ctl.sensitivity = 1.0
        self.ctl.apply(_frame(0.75, 0.5, events=[_event(CURSOR_MOVE)]), now=1.2)
        near = self.mouse.pos[0]
        self.assertGreater(far, near)

    def test_short_pinch_clicks_once(self):
        self.ctl.enable()
        self.ctl.apply(
            _frame(pinching=True, events=[_event(CURSOR_MOVE, now=1.00), _event(PINCH_START, now=1.00)]),
            now=1.00,
        )
        self.ctl.apply(
            _frame(pinching=True, events=[_event(CURSOR_MOVE, now=1.08), _event(PINCH_HOLD, now=1.08)]),
            now=1.08,
        )
        self.assertNotIn("down", self._kinds())
        self.ctl.apply(
            _frame(pinching=False, events=[_event(CURSOR_MOVE, now=1.16), _event(PINCH_END, now=1.16)]),
            now=1.16,
        )
        self.assertEqual(self._kinds()[-2:], ["down", "up"])
        self.assertFalse(self.mouse.pressed)
        self.assertEqual(self._kinds().count("down"), 1)

    def test_held_pinch_drags_without_click(self):
        self.ctl.enable()
        self.ctl.apply(
            _frame(0.4, 0.4, pinching=True, events=[_event(PINCH_START, 0.4, 0.4, 2.00)]),
            now=2.00,
        )
        self.ctl.apply(
            _frame(0.4, 0.4, pinching=True, events=[_event(PINCH_HOLD, 0.4, 0.4, 2.10)]),
            now=2.10,
        )
        self.assertFalse(self.mouse.pressed)
        self.ctl.apply(
            _frame(0.55, 0.4, pinching=True, events=[_event(PINCH_HOLD, 0.55, 0.4, 2.30)]),
            now=2.30,
        )
        self.assertTrue(self.mouse.pressed)
        self.assertIn("down", self._kinds())
        self.assertNotIn("up", self._kinds())
        self.ctl.apply(
            _frame(0.60, 0.4, pinching=False, events=[_event(PINCH_END, 0.60, 0.4, 2.40)]),
            now=2.40,
        )
        self.assertFalse(self.mouse.pressed)
        self.assertEqual(self._kinds().count("down"), 1)
        self.assertEqual(self._kinds().count("up"), 1)

    def test_lost_tracking_releases_and_does_not_click(self):
        self.ctl.enable()
        self.ctl.apply(
            _frame(pinching=True, events=[_event(PINCH_START, now=3.00)]),
            now=3.00,
        )
        self.ctl.apply(
            _frame(
                visible=False,
                events=[_event(PINCH_END, now=3.05)],
            ),
            now=3.05,
        )
        self.assertNotIn("down", self._kinds())
        self.assertFalse(self.mouse.pressed)
        self.assertTrue(self.ctl.enabled)
        self.assertEqual(self.ctl.status, "LOST")

    def test_lost_tracking_auto_disengages(self):
        self.ctl.enable()
        self.ctl.apply(_frame(visible=False), now=4.00)
        self.assertTrue(self.ctl.enabled)
        self.ctl.apply(_frame(visible=False), now=4.60)
        self.assertFalse(self.ctl.enabled)
        self.assertEqual(self.stopped[-1], "TRACKING LOST")

    def test_lost_while_dragging_releases_button(self):
        self.ctl.enable()
        self.ctl.apply(
            _frame(pinching=True, events=[_event(PINCH_START, now=5.00)]),
            now=5.00,
        )
        self.ctl.apply(
            _frame(pinching=True, events=[_event(PINCH_HOLD, now=5.30)]),
            now=5.30,
        )
        self.assertTrue(self.mouse.pressed)
        self.ctl.apply(_frame(visible=False), now=5.35)
        self.assertFalse(self.mouse.pressed)
        self.assertIn("up", self._kinds())

    def test_disable_releases_held_button(self):
        self.ctl.enable()
        self.ctl.apply(
            _frame(pinching=True, events=[_event(PINCH_START, now=6.00)]),
            now=6.00,
        )
        self.ctl.apply(
            _frame(pinching=True, events=[_event(PINCH_HOLD, now=6.30)]),
            now=6.30,
        )
        self.assertTrue(self.mouse.pressed)
        self.ctl.disable(reason="DISABLED")
        self.assertFalse(self.mouse.pressed)
        self.assertFalse(self.ctl.enabled)
        self.assertEqual(self.stopped[-1], "DISABLED")

    def test_emergency_stop(self):
        self.ctl.enable()
        self.ctl.emergency_stop()
        self.assertFalse(self.ctl.enabled)
        self.assertEqual(self.stopped[-1], "EMERGENCY STOP")


if __name__ == "__main__":
    unittest.main()
