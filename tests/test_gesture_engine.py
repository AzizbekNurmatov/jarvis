"""Gesture engine one-shot recognition tests (no camera required)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from aura.core.gesture_engine import GestureEngine
from aura.core.pose import wrap_angle, wrap_angles
import numpy as np


def blink(label, detected=True):
    return SimpleNamespace(latest=label, detected=detected)


class GestureEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = GestureEngine(
            double_blink_window_s=0.50,
            both_defer_s=0.20,
            nod_down_deg=12.0,
            nod_return_deg=6.0,
            nod_window_s=0.85,
            shake_deg=12.0,
            shake_return_deg=6.0,
            shake_window_s=1.1,
            turn_deg=18.0,
            turn_hold_s=0.20,
            turn_return_deg=10.0,
            cooldown_s=0.05,
            head_cooldown_s=0.05,
        )

    def test_left_wink_gesture(self):
        event = self.engine.update(
            blink("Left Eye"), [0.0, 0.0, 0.0], now=1.0
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.name, "LEFT_WINK")
        self.assertEqual(event.label, "LEFT WINK")

    def test_double_blink_not_both(self):
        first = self.engine.update(
            blink("Both Eyes"), [0, 0, 0], now=1.00
        )
        self.assertIsNone(first)

        # Still within defer window: waiting for possible double.
        still_pending = self.engine.update(
            blink("Both Eyes", detected=False),
            [0, 0, 0],
            now=1.10,
        )
        self.assertIsNone(still_pending)

        second = self.engine.update(
            blink("Both Eyes"), [0, 0, 0], now=1.25
        )
        self.assertIsNotNone(second)
        self.assertEqual(second.name, "DOUBLE_BLINK")

    def test_single_both_blink_after_defer(self):
        pending = self.engine.update(
            blink("Both Eyes"), [0, 0, 0], now=2.00
        )
        self.assertIsNone(pending)

        event = self.engine.update(
            blink("Both Eyes", detected=False),
            [0, 0, 0],
            now=2.25,
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.name, "BOTH_BLINK")

    def test_head_nod(self):
        self.engine.update(None, [0.0, 0.0, 0.0], now=5.0)
        self.engine.update(None, [0.0, 16.0, 0.0], now=5.1)
        event = self.engine.update(
            None, [0.0, 2.0, 0.0], now=5.4
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.name, "HEAD_NOD")

    def test_head_shake(self):
        self.engine.update(None, [0.0, 0.0, 0.0], now=6.0)
        self.engine.update(None, [-16.0, 0.0, 0.0], now=6.1)
        self.engine.update(None, [16.0, 0.0, 0.0], now=6.3)
        event = self.engine.update(
            None, [1.0, 0.0, 0.0], now=6.5
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.name, "HEAD_SHAKE")

    def test_head_turn_left_hold(self):
        self.engine.update(None, [-20.0, 0.0, 0.0], now=7.0)
        none_yet = self.engine.update(
            None, [-22.0, 0.0, 0.0], now=7.1
        )
        self.assertIsNone(none_yet)
        event = self.engine.update(
            None, [-22.0, 0.0, 0.0], now=7.25
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.name, "HEAD_TURN_LEFT")

    def test_one_shot_cooldown(self):
        first = self.engine.update(
            blink("Right Eye"), [0, 0, 0], now=8.0
        )
        self.assertEqual(first.name, "RIGHT_WINK")
        second = self.engine.update(
            blink("Right Eye"), [0, 0, 0], now=8.02
        )
        self.assertIsNone(second)


class AngleWrapTests(unittest.TestCase):
    def test_wrap_near_full_circle(self):
        self.assertAlmostEqual(wrap_angle(356.0), -4.0, places=5)
        self.assertAlmostEqual(wrap_angle(-190.0), 170.0, places=5)

    def test_relative_pitch_no_356(self):
        absolute = np.array([10.0, 358.0, 0.0])
        neutral = np.array([8.0, 2.0, 0.0])
        relative = wrap_angles(absolute - neutral)
        self.assertAlmostEqual(relative[1], -4.0, places=5)
        self.assertLess(abs(relative[1]), 20.0)


if __name__ == "__main__":
    unittest.main()
