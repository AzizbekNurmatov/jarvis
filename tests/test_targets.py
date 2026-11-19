"""Image-plane target aim and pinch-to-confirm tests (no camera)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from aura.core.targets import (
    AIM_MODE,
    STATE_AMBIGUOUS,
    STATE_AIMING,
    STATE_CALIBRATE,
    STATE_CONFIRMED,
    STATE_IDLE,
    STATE_LOCKED,
    STATE_LOST,
    TargetSelector,
    resolve_aim,
)


def _hand(mcp, tip):
    points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    points[5] = SimpleNamespace(x=mcp[0], y=mcp[1])
    points[8] = SimpleNamespace(x=tip[0], y=tip[1])
    return points


# Finger points to the right. The box sits in front of the tip.
_AHEAD = _hand((0.30, 0.50), (0.45, 0.50))
_BOX = (0.60, 0.40, 0.20, 0.20)


class TargetSelectionTests(unittest.TestCase):
    def _selector(self, **kwargs):
        settings = {"stable_s": 0.40, "confirm_cooldown_s": 0.85}
        settings.update(kwargs)
        selector = TargetSelector(**settings)
        region = selector.add_region("Lamp", *_BOX)
        self.assertIsNotNone(region)
        return selector

    def test_rejects_a_fourth_region_and_a_tiny_one(self):
        selector = TargetSelector()
        for index in range(3):
            self.assertIsNotNone(
                selector.add_region(f"T{index}", 0.1 * index, 0.1, 0.08, 0.08)
            )
        self.assertIsNone(selector.add_region("Extra", 0.5, 0.5, 0.1, 0.1))
        self.assertIsNone(selector.add_region("Tiny", 0.2, 0.2, 0.01, 0.2))

    def test_ray_hits_the_region_in_front_not_behind(self):
        selector = self._selector()
        ahead = resolve_aim(_AHEAD, selector.regions, aspect=1.0)
        self.assertEqual(ahead.region.name, "Lamp")
        self.assertFalse(ahead.ambiguous)
        self.assertGreater(ahead.ray[2], ahead.ray[0])

        behind = selector
        behind.regions.clear()
        behind.add_region("Back", 0.05, 0.40, 0.10, 0.20)
        missed = resolve_aim(_AHEAD, behind.regions, aspect=1.0)
        self.assertIsNone(missed.region)
        self.assertIsNotNone(missed.ray)

    def test_tip_inside_a_region_aims_it(self):
        selector = TargetSelector()
        selector.add_region("Lamp", *_BOX)
        inside = _hand((0.50, 0.50), (0.70, 0.50))
        hit = resolve_aim(inside, selector.regions, aspect=1.0)
        self.assertEqual(hit.region.name, "Lamp")

    def test_overlapping_regions_are_ambiguous(self):
        selector = TargetSelector()
        selector.add_region("A", 0.40, 0.40, 0.30, 0.30)
        selector.add_region("B", 0.45, 0.45, 0.30, 0.30)
        frame = selector.update(
            _hand((0.40, 0.55), (0.55, 0.55)),
            now=1.0,
            pointing_stable=True,
            pointing_raw=True,
        )
        self.assertEqual(frame.state, STATE_AMBIGUOUS)
        self.assertIsNone(frame.selection)
        self.assertEqual(frame.aimed_name, "")
        self.assertEqual(len(frame.contenders), 2)

    def test_stable_point_then_pinch_confirms_once(self):
        selector = self._selector()
        aiming = selector.update(_AHEAD, now=1.00, pointing_raw=True, pointing_stable=True)
        self.assertEqual(aiming.state, STATE_AIMING)
        self.assertIsNone(aiming.selection)
        locked = selector.update(_AHEAD, now=1.45, pointing_raw=True, pointing_stable=True)
        self.assertEqual(locked.state, STATE_LOCKED)
        confirmed = selector.update(_AHEAD, now=1.55, pinch_started=True)
        self.assertEqual(confirmed.state, STATE_CONFIRMED)
        self.assertEqual(confirmed.selection.name, "Lamp")
        self.assertEqual(confirmed.selection.mode, AIM_MODE)
        self.assertEqual(confirmed.mode, AIM_MODE)
        self.assertIs(confirmed.last_selection, confirmed.selection)

    def test_pinch_before_the_point_is_stable_does_not_select(self):
        selector = self._selector()
        selector.update(_AHEAD, now=1.00, pointing_raw=True, pointing_stable=True)
        early = selector.update(_AHEAD, now=1.20, pinch_started=True)
        self.assertIsNone(early.selection)
        self.assertEqual(early.state, STATE_IDLE)
        self.assertIsNone(early.last_selection)

    def test_cooldown_blocks_a_second_pinch(self):
        selector = self._selector(flash_s=0.0, confirm_cooldown_s=2.0)
        selector.update(_AHEAD, now=1.00, pointing_stable=True, pointing_raw=True)
        selector.update(_AHEAD, now=1.45, pointing_stable=True, pointing_raw=True)
        first = selector.update(_AHEAD, now=1.50, pinch_started=True)
        selector.update(_AHEAD, now=1.60, pointing_stable=True, pointing_raw=True)
        selector.update(_AHEAD, now=2.05, pointing_stable=True, pointing_raw=True)
        second = selector.update(_AHEAD, now=2.10, pinch_started=True)
        self.assertEqual(first.selection.name, "Lamp")
        self.assertIsNone(second.selection)
        self.assertEqual(second.last_selection.name, "Lamp")

    def test_switching_targets_resets_the_dwell(self):
        selector = self._selector()
        selector.add_region("Book", 0.10, 0.10, 0.20, 0.20)
        selector.update(_AHEAD, now=1.00, pointing_stable=True, pointing_raw=True)
        selector.update(_AHEAD, now=1.45, pointing_stable=True, pointing_raw=True)
        other = _hand((0.40, 0.20), (0.25, 0.20))
        moved = selector.update(other, now=1.55, pointing_stable=True, pointing_raw=True)
        self.assertEqual(moved.state, STATE_AIMING)
        self.assertEqual(moved.aimed_name, "Book")
        pinched = selector.update(other, now=1.70, pinch_started=True)
        self.assertIsNone(pinched.selection)

    def test_tracking_loss_keeps_the_last_selection(self):
        selector = self._selector()
        selector.update(_AHEAD, now=1.00, pointing_stable=True, pointing_raw=True)
        selector.update(_AHEAD, now=1.45, pointing_stable=True, pointing_raw=True)
        selector.update(_AHEAD, now=1.55, pinch_started=True)
        lost = selector.update(None, now=1.80)
        self.assertEqual(lost.state, STATE_LOST)
        self.assertEqual(lost.last_selection.name, "Lamp")
        self.assertIsNone(lost.selection)
        stray = selector.update(_AHEAD, now=1.90, pinch_started=True)
        self.assertIsNone(stray.selection)
        self.assertEqual(stray.last_selection.name, "Lamp")

    def test_open_hand_does_not_select(self):
        selector = self._selector()
        frame = selector.update(_AHEAD, now=1.0, pointing_stable=False, pointing_raw=False)
        self.assertEqual(frame.state, STATE_IDLE)
        self.assertIsNone(frame.selection)

    def test_no_regions_asks_for_calibration(self):
        selector = TargetSelector()
        frame = selector.update(_AHEAD, now=1.0, pointing_stable=True, pointing_raw=True)
        self.assertEqual(frame.state, STATE_CALIBRATE)
        self.assertIsNone(frame.selection)


if __name__ == "__main__":
    unittest.main()
