"""Interaction engine cursor + pinch event tests (no camera / Tk required)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from aura.core.interaction import (
    CURSOR_MOVE,
    PINCH_END,
    PINCH_HOLD,
    PINCH_START,
    InteractionEngine,
)
from tests.test_hand_gestures import hand_obj, open_palm, pinch, pointing


def _events(frame, name):
    return [e for e in frame.events if e.name == name]


class InteractionEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = InteractionEngine(
            smooth=1.0,
            start_frames=2,
            end_frames=2,
        )

    def test_no_hand_hides_cursor(self):
        frame = self.engine.update([], now=1.0)
        self.assertFalse(frame.visible)
        self.assertFalse(frame.pinching)
        self.assertEqual(frame.events, [])

    def test_index_tip_maps_to_cursor(self):
        frame = self.engine.update(
            [hand_obj(pointing(), "Right")], now=1.0
        )
        self.assertTrue(frame.visible)
        self.assertEqual(frame.hand, "RIGHT")
        self.assertTrue(0.0 <= frame.cursor_x <= 1.0)
        self.assertTrue(0.0 <= frame.cursor_y <= 1.0)
        self.assertEqual(len(_events(frame, CURSOR_MOVE)), 1)

    def test_smoothing_moves_toward_target(self):
        engine = InteractionEngine(smooth=0.5, start_frames=2, end_frames=2)
        first = engine.update([hand_obj(open_palm(), "Right")], now=1.0)
        x0 = first.cursor_x
        moved = pointing()
        moved[8].x = min(0.95, moved[8].x + 0.20)
        unsmoothed = InteractionEngine(smooth=1.0)
        target = unsmoothed.update([hand_obj(moved, "Right")], now=1.0)
        second = engine.update([hand_obj(moved, "Right")], now=1.1)
        self.assertGreater(second.cursor_x, x0)
        self.assertLess(second.cursor_x, target.cursor_x)

    def test_pinch_start_hold_end(self):
        self.engine.update([hand_obj(pointing(), "Right")], now=1.00)
        first = self.engine.update([hand_obj(pinch(), "Right")], now=1.05)
        self.assertEqual(_events(first, PINCH_START), [])
        self.assertFalse(first.pinching)

        started = self.engine.update([hand_obj(pinch(), "Right")], now=1.12)
        self.assertEqual(len(_events(started, PINCH_START)), 1)
        self.assertTrue(started.pinching)

        held = self.engine.update([hand_obj(pinch(), "Right")], now=1.20)
        self.assertEqual(len(_events(held, PINCH_HOLD)), 1)
        self.assertEqual(_events(held, PINCH_START), [])

        self.engine.update([hand_obj(pointing(), "Right")], now=1.28)
        ended = self.engine.update([hand_obj(pointing(), "Right")], now=1.36)
        self.assertEqual(len(_events(ended, PINCH_END)), 1)
        self.assertFalse(ended.pinching)

    def test_lost_hand_cancels_pinch(self):
        self.engine.update([hand_obj(pinch(), "Right")], now=2.00)
        self.engine.update([hand_obj(pinch(), "Right")], now=2.10)
        lost = self.engine.update([], now=2.20)
        self.assertFalse(lost.visible)
        self.assertEqual(len(_events(lost, PINCH_END)), 1)

    def test_pinch_does_not_retrigger_while_held(self):
        self.engine.update([hand_obj(pinch(), "Left")], now=3.00)
        self.engine.update([hand_obj(pinch(), "Left")], now=3.08)
        starts = 0
        holds = 0
        for i in range(5):
            frame = self.engine.update(
                [hand_obj(pinch(), "Left")], now=3.16 + i * 0.08
            )
            starts += len(_events(frame, PINCH_START))
            holds += len(_events(frame, PINCH_HOLD))
        self.assertEqual(starts, 0)
        self.assertEqual(holds, 5)

    def test_reset_clears_state(self):
        self.engine.update([hand_obj(pinch(), "Right")], now=4.00)
        self.engine.update([hand_obj(pinch(), "Right")], now=4.10)
        self.engine.reset()
        frame = self.engine.update([], now=4.20)
        self.assertEqual(frame.events, [])
        self.assertFalse(frame.visible)


if __name__ == "__main__":
    unittest.main()
