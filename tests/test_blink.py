"""Blink / wink classification tests (no camera required)."""

from __future__ import annotations

import unittest

from aura.core.blink import BlinkDetector


class BlinkClassificationTests(unittest.TestCase):
    def setUp(self):
        self.detector = BlinkDetector(
            close_ratio=0.82,
            open_ratio=0.90,
            wink_partner_ratio=0.88,
            min_closed_frames=1,
            reopen_frames=1,
            sync_window_s=0.16,
            settle_s=0.0,
            cooldown_s=0.0,
        )

    def _run(self, samples):
        """samples: list of (t, left_ear, right_ear)."""
        events = []
        for now, left, right in samples:
            result = self.detector.update_from_ears(
                left, right, now=now
            )
            if result.detected:
                events.append(result.latest)
        return events

    def test_left_wink_independent(self):
        open_l = open_r = 0.30
        # Establish baselines, then close only left.
        samples = [
            (0.00, open_l, open_r),
            (0.03, open_l, open_r),
            (0.06, 0.18, open_r),
            (0.09, 0.16, open_r),
            (0.12, open_l, open_r),
            (0.15, open_l, open_r),
        ]
        events = self._run(samples)
        self.assertEqual(events, ["Left Eye"])

    def test_right_wink_independent(self):
        open_l = open_r = 0.30
        samples = [
            (0.00, open_l, open_r),
            (0.03, open_l, open_r),
            (0.06, open_l, 0.17),
            (0.09, open_l, 0.15),
            (0.12, open_l, open_r),
            (0.15, open_l, open_r),
        ]
        events = self._run(samples)
        self.assertEqual(events, ["Right Eye"])

    def test_both_eyes_near_simultaneous(self):
        open_l = open_r = 0.30
        samples = [
            (0.00, open_l, open_r),
            (0.03, open_l, open_r),
            (0.06, 0.16, 0.17),
            (0.09, 0.15, 0.15),
            (0.12, open_l, open_r),
            (0.15, open_l, open_r),
        ]
        events = self._run(samples)
        self.assertEqual(events, ["Both Eyes"])

    def test_no_duplicate_from_single_blink(self):
        open_l = open_r = 0.30
        samples = [
            (0.00, open_l, open_r),
            (0.02, 0.16, 0.16),
            (0.04, 0.15, 0.15),
            (0.06, open_l, open_r),
            (0.08, open_l, open_r),
            (0.10, open_l, open_r),
            (0.12, open_l, open_r),
        ]
        events = self._run(samples)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], "Both Eyes")

    def test_openness_tracks_ear(self):
        result = self.detector.update_from_ears(0.30, 0.30, now=0.0)
        self.assertGreaterEqual(result.left_openness, 0.95)
        closed = self.detector.update_from_ears(0.10, 0.30, now=0.05)
        self.assertLess(closed.left_openness, 0.5)
        self.assertGreater(closed.right_openness, 0.9)


if __name__ == "__main__":
    unittest.main()
