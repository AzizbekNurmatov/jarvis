"""Interaction engine cursor + fist event tests (no camera / Tk required)."""

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
from tests.test_hand_gestures import fist, hand_obj, open_palm, pointing


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

    def test_fast_move_stays_close_to_the_finger(self):
        engine = InteractionEngine()
        start = pointing()
        engine.update([hand_obj(start, "Right")], now=1.0)
        moved = pointing()
        for lm in moved:
            lm.x += 0.18
        raw = InteractionEngine(smooth=1.0)
        target = raw.update([hand_obj(moved, "Right")], now=1.0)
        origin = raw.update([hand_obj(start, "Right")], now=2.0)
        # Two frames, like a real camera, so speed can open the filter.
        engine.update([hand_obj(moved, "Right")], now=1.0 + 1.0 / 30.0)
        caught = engine.update([hand_obj(moved, "Right")], now=1.0 + 2.0 / 30.0)
        span = abs(target.cursor_x - origin.cursor_x)
        self.assertGreater(span, 0.15)
        covered = abs(caught.cursor_x - origin.cursor_x)
        self.assertGreater(covered, span * 0.75)

    def test_resting_finger_rejects_jitter(self):
        engine = InteractionEngine()
        hand = pointing()
        t = 1.0
        engine.update([hand_obj(hand, "Right")], now=t)
        for _ in range(10):
            t += 1.0 / 30.0
            settled = engine.update([hand_obj(hand, "Right")], now=t)
        jitter = pointing()
        jitter[8].x = hand[8].x + 0.012
        jitter[8].y = hand[8].y - 0.010
        t += 1.0 / 30.0
        shaken = engine.update([hand_obj(jitter, "Right")], now=t)
        self.assertLess(abs(shaken.cursor_x - settled.cursor_x), 0.006)
        self.assertLess(abs(shaken.cursor_y - settled.cursor_y), 0.006)

    def test_fist_start_hold_end(self):
        self.engine.update([hand_obj(pointing(), "Right")], now=1.00)
        first = self.engine.update([hand_obj(fist(), "Right")], now=1.05)
        self.assertEqual(_events(first, PINCH_START), [])
        self.assertFalse(first.pinching)

        started = self.engine.update([hand_obj(fist(), "Right")], now=1.12)
        self.assertEqual(len(_events(started, PINCH_START)), 1)
        self.assertTrue(started.pinching)

        held = self.engine.update([hand_obj(fist(), "Right")], now=1.20)
        self.assertEqual(len(_events(held, PINCH_HOLD)), 1)
        self.assertEqual(_events(held, PINCH_START), [])

        self.engine.update([hand_obj(pointing(), "Right")], now=1.28)
        ended = self.engine.update([hand_obj(pointing(), "Right")], now=1.36)
        self.assertEqual(len(_events(ended, PINCH_END)), 1)
        self.assertFalse(ended.pinching)

    def test_lost_hand_cancels_fist(self):
        self.engine.update([hand_obj(fist(), "Right")], now=2.00)
        self.engine.update([hand_obj(fist(), "Right")], now=2.10)
        lost = self.engine.update([], now=2.20)
        self.assertFalse(lost.visible)
        self.assertEqual(len(_events(lost, PINCH_END)), 1)

    def test_fist_does_not_retrigger_while_held(self):
        self.engine.update([hand_obj(fist(), "Left")], now=3.00)
        self.engine.update([hand_obj(fist(), "Left")], now=3.08)
        starts = 0
        holds = 0
        for i in range(5):
            frame = self.engine.update(
                [hand_obj(fist(), "Left")], now=3.16 + i * 0.08
            )
            starts += len(_events(frame, PINCH_START))
            holds += len(_events(frame, PINCH_HOLD))
        self.assertEqual(starts, 0)
        self.assertEqual(holds, 5)

    def test_reset_clears_state(self):
        self.engine.update([hand_obj(fist(), "Right")], now=4.00)
        self.engine.update([hand_obj(fist(), "Right")], now=4.10)
        self.engine.reset()
        frame = self.engine.update([], now=4.20)
        self.assertEqual(frame.events, [])
        self.assertFalse(frame.visible)


if __name__ == "__main__":
    unittest.main()
