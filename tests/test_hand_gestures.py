"""Hand gesture classification + stabilization tests (no camera required)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from aura.core.hand_gestures import (
    EVENT_ENDED,
    EVENT_HELD,
    EVENT_STARTED,
    FIST,
    NO_HAND,
    OPEN_PALM,
    PEACE_SIGN,
    PINCH,
    POINTING,
    STATUS_DETECTING,
    STATUS_NO_HAND,
    STATUS_STABLE,
    UNKNOWN,
    HandGestureRecognizer,
    classify_landmarks,
)


class LM:
    def __init__(self, x, y, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def make_hand(
    index="extended",
    middle="extended",
    ring="extended",
    pinky="extended",
    thumb="extended",
    pinch=False,
    origin=(0.50, 0.78, 0.0),
):
    """Build 21 landmarks for a palm-facing hand (image y grows downward)."""
    pts = [None] * 21
    wrist = origin
    pts[0] = wrist

    # Palm MCPs sit above the wrist.
    pts[5] = _add(wrist, (-0.07, -0.20, 0.0))   # index MCP
    pts[9] = _add(wrist, (-0.01, -0.22, 0.0))   # middle MCP
    pts[13] = _add(wrist, (0.05, -0.21, 0.0))   # ring MCP
    pts[17] = _add(wrist, (0.10, -0.18, 0.0))   # pinky MCP

    # Thumb chain along the radial side of the palm.
    pts[1] = _add(wrist, (-0.04, -0.05, 0.0))   # CMC
    pts[2] = _add(wrist, (-0.09, -0.10, 0.0))   # MCP

    def place_finger(mcp_idx, pip_idx, dip_idx, tip_idx, direction, length, state):
        mcp = pts[mcp_idx]
        if state == "extended":
            pts[pip_idx] = _add(mcp, _scale(direction, length * 0.38))
            pts[dip_idx] = _add(mcp, _scale(direction, length * 0.70))
            pts[tip_idx] = _add(mcp, _scale(direction, length))
        elif state == "pinch":
            # Bent but still distal — typical pinch grip, not a folded fist.
            pts[pip_idx] = _add(mcp, _scale(direction, length * 0.32))
            pts[dip_idx] = _add(mcp, _scale(direction, length * 0.52))
            pts[tip_idx] = _add(mcp, _scale(direction, length * 0.68))
        else:
            pts[pip_idx] = _add(mcp, _scale(direction, length * 0.26))
            pts[dip_idx] = _add(mcp, _scale(direction, length * 0.06))
            pts[tip_idx] = _add(mcp, _scale(direction, -length * 0.16))

    place_finger(5, 6, 7, 8, (-0.08, -1.0, 0.0), 0.22, index)
    place_finger(9, 10, 11, 12, (0.0, -1.0, 0.0), 0.24, middle)
    place_finger(13, 14, 15, 16, (0.08, -1.0, 0.0), 0.22, ring)
    place_finger(17, 18, 19, 20, (0.16, -1.0, 0.0), 0.18, pinky)

    if pinch:
        index_tip = pts[8]
        # Thumb tip meets the index tip; IP sits between MCP and tip.
        pts[4] = index_tip
        pts[3] = (
            (pts[2][0] + index_tip[0]) / 2.0,
            (pts[2][1] + index_tip[1]) / 2.0,
            0.0,
        )
    elif thumb == "extended":
        pts[3] = _add(pts[2], (-0.07, -0.06, 0.0))
        pts[4] = _add(pts[2], (-0.14, -0.12, 0.0))
    else:
        pts[3] = _add(pts[2], (0.02, -0.03, 0.0))
        pts[4] = _add(pts[9], (-0.02, 0.04, 0.0))

    return [LM(*p) for p in pts]


def hand_obj(landmarks, label="Right"):
    return SimpleNamespace(label=label, landmarks=landmarks)


def open_palm():
    return make_hand()


def fist():
    return make_hand(
        index="curled",
        middle="curled",
        ring="curled",
        pinky="curled",
        thumb="curled",
    )


def pointing():
    return make_hand(
        index="extended",
        middle="curled",
        ring="curled",
        pinky="curled",
        thumb="curled",
    )


def peace():
    return make_hand(
        index="extended",
        middle="extended",
        ring="curled",
        pinky="curled",
        thumb="curled",
    )


def pinch():
    return make_hand(
        index="extended",
        middle="curled",
        ring="curled",
        pinky="curled",
        thumb="extended",
        pinch=True,
    )


class ClassifyLandmarksTests(unittest.TestCase):
    def test_open_palm(self):
        self.assertEqual(classify_landmarks(open_palm()), OPEN_PALM)

    def test_fist(self):
        self.assertEqual(classify_landmarks(fist()), FIST)

    def test_pointing(self):
        self.assertEqual(classify_landmarks(pointing()), POINTING)

    def test_peace_sign(self):
        self.assertEqual(classify_landmarks(peace()), PEACE_SIGN)

    def test_pinch(self):
        self.assertEqual(classify_landmarks(pinch()), PINCH)

    def test_pinch_beats_pointing(self):
        # Index out + tips together must not be reported as POINTING.
        pose = make_hand(
            index="extended",
            middle="curled",
            ring="curled",
            pinky="curled",
            thumb="extended",
            pinch=True,
        )
        self.assertEqual(classify_landmarks(pose), PINCH)

    def test_bent_index_pinch_not_fist(self):
        pose = make_hand(
            index="pinch",
            middle="curled",
            ring="curled",
            pinky="curled",
            thumb="extended",
            pinch=True,
        )
        self.assertEqual(classify_landmarks(pose), PINCH)

    def test_left_and_right_same_pose(self):
        landmarks = pointing()
        self.assertEqual(classify_landmarks(landmarks), POINTING)
        mirrored = make_hand(
            index="extended",
            middle="curled",
            ring="curled",
            pinky="curled",
            thumb="curled",
            origin=(0.35, 0.80, 0.0),
        )
        self.assertEqual(classify_landmarks(mirrored), POINTING)

    def test_three_fingers_unknown(self):
        pose = make_hand(
            index="extended",
            middle="extended",
            ring="extended",
            pinky="curled",
            thumb="curled",
        )
        self.assertEqual(classify_landmarks(pose), UNKNOWN)

    def test_missing_landmarks_unknown(self):
        self.assertEqual(classify_landmarks(None), UNKNOWN)
        self.assertEqual(classify_landmarks([]), UNKNOWN)


class StabilizerTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = HandGestureRecognizer(
            confirm_s=0.05,
            min_frames=2,
            history_frames=5,
            max_log=5,
        )

    def _step(self, landmarks, label="Right", now=1.0):
        return self.recognizer.update(
            [hand_obj(landmarks, label)], now=now
        )

    def test_no_hand(self):
        snap = self.recognizer.update([], now=1.0)
        self.assertEqual(snap.primary.gesture, NO_HAND)
        self.assertEqual(snap.primary.status, STATUS_NO_HAND)
        self.assertEqual(snap.hands, [])

    def test_confirms_after_hold(self):
        first = self._step(open_palm(), now=1.00)
        self.assertNotEqual(first.primary.event, EVENT_STARTED)
        self.assertEqual(first.primary.status, STATUS_DETECTING)
        self.assertEqual(first.primary.hand, "RIGHT")

        second = self._step(open_palm(), now=1.08)
        self.assertEqual(second.primary.gesture, OPEN_PALM)
        self.assertEqual(second.primary.status, STATUS_STABLE)
        self.assertEqual(second.primary.event, EVENT_STARTED)
        self.assertEqual(second.primary.hand, "RIGHT")
        self.assertEqual(second.primary.label, "OPEN PALM")

    def test_flicker_does_not_confirm(self):
        self._step(open_palm(), now=1.00)
        flicker = self._step(fist(), now=1.03)
        self.assertNotEqual(flicker.primary.gesture, FIST)
        self.assertNotEqual(flicker.primary.event, EVENT_STARTED)

    def test_held_does_not_relog(self):
        self._step(pointing(), now=2.00)
        self._step(pointing(), now=2.08)
        third = self._step(pointing(), now=2.16)
        self.assertEqual(third.primary.event, EVENT_HELD)
        self.assertEqual(len(self.recognizer.recent), 1)
        self.assertEqual(self.recognizer.recent[0].gesture, POINTING)

    def test_hand_lost_emits_ended_and_clears(self):
        self._step(peace(), now=3.00)
        self._step(peace(), now=3.08)
        lost = self.recognizer.update([], now=3.20)
        self.assertEqual(lost.primary.gesture, NO_HAND)
        self.assertEqual(lost.primary.status, STATUS_NO_HAND)
        self.assertEqual(lost.primary.event, EVENT_ENDED)
        self.assertEqual(lost.primary.hand, "")

    def test_hands_are_independent(self):
        self.recognizer.update(
            [
                hand_obj(open_palm(), "Left"),
                hand_obj(fist(), "Right"),
            ],
            now=4.00,
        )
        snap = self.recognizer.update(
            [
                hand_obj(open_palm(), "Left"),
                hand_obj(fist(), "Right"),
            ],
            now=4.10,
        )
        by_hand = {r.hand: r.gesture for r in snap.hands}
        self.assertEqual(by_hand["LEFT"], OPEN_PALM)
        self.assertEqual(by_hand["RIGHT"], FIST)
        self.assertEqual(len(snap.recent), 2)
        logged = {e.hand: e.gesture for e in snap.recent}
        self.assertEqual(logged["LEFT"], OPEN_PALM)
        self.assertEqual(logged["RIGHT"], FIST)

    def test_reset_clears_state_and_log(self):
        self._step(pinch(), now=5.00)
        self._step(pinch(), now=5.10)
        self.assertTrue(self.recognizer.recent)
        self.recognizer.reset()
        snap = self.recognizer.update([], now=5.20)
        self.assertEqual(snap.primary.gesture, NO_HAND)
        self.assertEqual(self.recognizer.recent, [])

    def test_primary_prefers_latest_named(self):
        self.recognizer.update(
            [hand_obj(open_palm(), "Left")], now=6.00
        )
        self.recognizer.update(
            [
                hand_obj(open_palm(), "Left"),
                hand_obj(pointing(), "Right"),
            ],
            now=6.10,
        )
        snap = self.recognizer.update(
            [
                hand_obj(open_palm(), "Left"),
                hand_obj(pointing(), "Right"),
            ],
            now=6.20,
        )
        self.assertEqual(snap.primary.hand, "RIGHT")
        self.assertEqual(snap.primary.gesture, POINTING)


if __name__ == "__main__":
    unittest.main()
