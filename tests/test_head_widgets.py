"""Head-anchored widget focus, selection, and placement tests."""

from __future__ import annotations

import unittest
from datetime import datetime
from types import SimpleNamespace

from aura.hud.head_widgets import (
    CLOCK,
    COUNTER,
    DATE,
    HeadWidgets,
    face_stack_origin,
)


def _event(name, now, direction=""):
    return SimpleNamespace(name=name, timestamp=now, direction=direction)


class _Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


def _face(cx=0.30, cy=0.50, fw=0.20, fh=0.40):
    points = [_Point(0.5, 0.5) for _ in range(468)]
    points[234] = _Point(cx - fw / 2.0, cy)
    points[454] = _Point(cx + fw / 2.0, cy)
    points[10] = _Point(cx, cy - fh / 2.0)
    points[152] = _Point(cx, cy + fh / 2.0)
    return points


class HeadWidgetTests(unittest.TestCase):
    def setUp(self):
        self.when = datetime(2026, 11, 17, 14, 5, 9)
        self.widgets = HeadWidgets(
            select_cooldown_s=0.70,
            clock=lambda: self.when,
        )

    def test_nod_cycles_without_activating(self):
        self.widgets.handle(_event("HEAD_NOD", 1.0, "down"))
        self.widgets.handle(_event("HEAD_NOD", 2.0, "down"))
        self.widgets.handle(_event("HEAD_NOD", 3.0, "down"))
        self.assertEqual(self.widgets.focus, CLOCK)
        self.assertEqual(self.widgets.active, [False, False, False])
        self.assertEqual(self.widgets.count, 0)

    def test_nod_up_cycles_the_other_way(self):
        self.widgets.handle(_event("HEAD_NOD", 1.0, "up"))
        self.assertEqual(self.widgets.focus, COUNTER)
        self.widgets.handle(_event("HEAD_NOD", 2.0, "down"))
        self.assertEqual(self.widgets.focus, CLOCK)

    def test_missing_direction_moves_down(self):
        self.widgets.handle(_event("HEAD_NOD", 1.0))
        self.assertEqual(self.widgets.focus, DATE)

    def test_double_blink_toggles_only_the_focused_widget(self):
        self.widgets.handle(_event("HEAD_NOD", 1.0, "down"))
        self.widgets.handle(_event("DOUBLE_BLINK", 2.0))
        self.assertEqual(self.widgets.focus, DATE)
        self.assertEqual(self.widgets.active, [False, True, False])
        self.assertEqual(self.widgets.value(DATE), "Nov 17")
        self.assertEqual(self.widgets.value(CLOCK), "Clock")
        self.assertEqual(self.widgets.count, 0)

        self.widgets.handle(_event("DOUBLE_BLINK", 3.0))
        self.assertFalse(self.widgets.active[DATE])
        self.assertEqual(self.widgets.value(DATE), "Date")

    def test_selection_cooldown_ignores_a_second_blink(self):
        self.widgets.handle(_event("DOUBLE_BLINK", 1.00))
        self.assertTrue(self.widgets.active[CLOCK])
        self.assertEqual(self.widgets.value(CLOCK), "14:05:09")
        self.widgets.handle(_event("DOUBLE_BLINK", 1.40))
        self.assertTrue(self.widgets.active[CLOCK])
        self.widgets.handle(_event("DOUBLE_BLINK", 1.80))
        self.assertFalse(self.widgets.active[CLOCK])

    def test_counter_increments_only_when_activated(self):
        self.widgets.handle(_event("HEAD_NOD", 1.0, "up"))
        self.assertEqual(self.widgets.focus, COUNTER)
        self.widgets.handle(_event("DOUBLE_BLINK", 2.0))
        self.assertEqual(self.widgets.count, 1)
        self.assertEqual(self.widgets.value(COUNTER), "1")
        self.widgets.handle(_event("DOUBLE_BLINK", 3.0))
        self.assertEqual(self.widgets.count, 1)
        self.assertEqual(self.widgets.value(COUNTER), "Counter")
        self.widgets.handle(_event("DOUBLE_BLINK", 4.0))
        self.assertEqual(self.widgets.count, 2)

    def test_nod_leaves_an_active_widget_on(self):
        self.widgets.handle(_event("DOUBLE_BLINK", 1.0))
        self.widgets.handle(_event("HEAD_NOD", 2.0, "down"))
        self.assertEqual(self.widgets.focus, DATE)
        self.assertTrue(self.widgets.active[CLOCK])
        self.assertFalse(self.widgets.active[DATE])
        self.assertEqual(self.widgets.count, 0)

    def test_other_gestures_do_nothing(self):
        self.widgets.handle(_event("BOTH_BLINK", 1.0))
        self.widgets.handle(_event("HEAD_SHAKE", 2.0))
        self.widgets.handle(None)
        self.assertEqual(self.widgets.focus, CLOCK)
        self.assertEqual(self.widgets.active, [False, False, False])

    def test_stack_sits_on_the_open_side_of_the_face(self):
        right_side = face_stack_origin(_face(cx=0.28), 640, 480)
        self.assertIsNotNone(right_side)
        face_right = (0.28 + 0.10) * 640
        self.assertGreater(right_side[0], face_right)

        left_side = face_stack_origin(_face(cx=0.78), 640, 480)
        face_left = (0.78 - 0.10) * 640
        self.assertLess(left_side[0] + left_side[2], face_left)

    def test_anchor_follows_without_jumping(self):
        widgets = HeadWidgets(follow_s=0.10)
        start = widgets._smooth_anchor((20.0, 40.0, 100.0, 36.0), 1.00)
        self.assertEqual(start[0], 20.0)
        moved = widgets._smooth_anchor((120.0, 40.0, 100.0, 36.0), 1.05)
        self.assertGreater(moved[0], 20.0)
        self.assertLess(moved[0], 120.0)
        snapped = widgets._smooth_anchor((10.0, 10.0, 80.0, 30.0), 2.00)
        self.assertEqual(snapped[0], 10.0)


if __name__ == "__main__":
    unittest.main()
