"""Independent per-eye blink / wink detection from MediaPipe Face Mesh."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time


@dataclass
class BlinkResult:
    latest: str = "None"
    left_closed: bool = False
    right_closed: bool = False
    left_ear: float = 0.0
    right_ear: float = 0.0
    left_openness: float = 1.0
    right_openness: float = 1.0
    detected: bool = False


class BlinkDetector:
    """
    MediaPipe Face Mesh blink detection.

    Labels refer to the person's anatomical left/right,
    not the left/right side of the displayed camera image.

    Only the latest result is retained; no blink history.
    """

    # Six landmarks around each eye (EAR).
    LEFT_EYE = (362, 385, 387, 263, 373, 380)
    RIGHT_EYE = (33, 160, 158, 133, 153, 144)

    # Full eyelid loops for the camera overlay (person's anatomical left/right).
    LEFT_EYE_CONTOUR = (
        362, 398, 384, 385, 386, 387, 388, 466, 263,
        249, 390, 373, 374, 380, 381, 382,
    )
    RIGHT_EYE_CONTOUR = (
        33, 246, 161, 160, 159, 158, 157, 173, 133,
        155, 154, 153, 145, 144, 163, 7,
    )

    def __init__(
        self,
        close_ratio=0.82,
        open_ratio=0.90,
        wink_partner_ratio=0.88,
        min_closed_frames=1,
        reopen_frames=1,
        sync_window_s=0.16,
        settle_s=0.05,
        cooldown_s=0.22,
        baseline_min=0.12,
    ):
        # Tunable thresholds (ratios relative to each eye's baseline EAR).
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.wink_partner_ratio = wink_partner_ratio
        self.min_closed_frames = min_closed_frames
        self.reopen_frames = reopen_frames
        self.sync_window_s = sync_window_s
        self.settle_s = settle_s
        self.cooldown_s = cooldown_s
        self.baseline_min = baseline_min

        self.baseline_left = None
        self.baseline_right = None

        self.latest = "None"

        self.left_closed = False
        self.right_closed = False
        self.left_frames = 0
        self.right_frames = 0
        self.left_open_frames = 0
        self.right_open_frames = 0

        self.episode_active = False
        self.episode_left = False
        self.episode_right = False
        self.episode_left_closed_at = None
        self.episode_right_closed_at = None
        self.episode_left_min_openness = 1.0
        self.episode_right_min_openness = 1.0
        self.both_open_since = None

        self.cooldown_until = 0.0

    @staticmethod
    def _distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _ear(self, landmarks, indices):
        points = [landmarks[i] for i in indices]

        vertical_1 = self._distance(points[1], points[5])
        vertical_2 = self._distance(points[2], points[4])
        horizontal = self._distance(points[0], points[3])

        if horizontal < 1e-6:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def _openness(self, ear, baseline):
        if baseline is None or baseline < 1e-6:
            return 1.0
        return max(0.0, min(1.25, ear / baseline))

    def reset(self):
        self.baseline_left = None
        self.baseline_right = None

        self.left_closed = False
        self.right_closed = False
        self.left_frames = 0
        self.right_frames = 0
        self.left_open_frames = 0
        self.right_open_frames = 0

        self._clear_episode()
        self.cooldown_until = 0.0
        self.latest = "None"

    def face_lost(self):
        # Keep the latest blink, but discard an incomplete event.
        self.left_closed = False
        self.right_closed = False
        self.left_frames = 0
        self.right_frames = 0
        self.left_open_frames = 0
        self.right_open_frames = 0
        self._clear_episode()

    def _clear_episode(self):
        self.episode_active = False
        self.episode_left = False
        self.episode_right = False
        self.episode_left_closed_at = None
        self.episode_right_closed_at = None
        self.episode_left_min_openness = 1.0
        self.episode_right_min_openness = 1.0
        self.both_open_since = None

    def _update_eye(self, ear, baseline, closed, close_frames, open_frames):
        close_threshold = baseline * self.close_ratio
        open_threshold = baseline * self.open_ratio

        if closed:
            if ear >= open_threshold:
                open_frames += 1
                if open_frames >= self.reopen_frames:
                    closed = False
                    close_frames = 0
                    open_frames = 0
            else:
                open_frames = 0
        else:
            if ear < close_threshold:
                close_frames += 1
                if close_frames >= self.min_closed_frames:
                    closed = True
                    open_frames = 0
            else:
                close_frames = 0

        return closed, close_frames, open_frames

    def _adapt_baseline(self, baseline, ear, closed):
        if closed or ear < baseline * self.open_ratio:
            return baseline
        # Learn open-eye EAR slowly; never collapse below a floor.
        updated = 0.97 * baseline + 0.03 * ear
        return max(updated, self.baseline_min)

    def _classify_episode(self):
        left = self.episode_left
        right = self.episode_right

        if not left and not right:
            return None

        if left and right:
            left_t = self.episode_left_closed_at
            right_t = self.episode_right_closed_at
            if left_t is not None and right_t is not None:
                if abs(left_t - right_t) <= self.sync_window_s:
                    return "Both Eyes"

            # Staggered closures: still both if both eyes participated.
            # Prefer wink only when the partner stayed clearly open.
            if (
                self.episode_right_min_openness >= self.wink_partner_ratio
                and self.episode_left_min_openness < self.close_ratio
            ):
                return "Left Eye"
            if (
                self.episode_left_min_openness >= self.wink_partner_ratio
                and self.episode_right_min_openness < self.close_ratio
            ):
                return "Right Eye"
            return "Both Eyes"

        if left:
            # Reject false wink if the other eye also dipped hard.
            if self.episode_right_min_openness < self.close_ratio:
                return "Both Eyes"
            if self.episode_right_min_openness < self.wink_partner_ratio:
                return "Both Eyes"
            return "Left Eye"

        if self.episode_left_min_openness < self.close_ratio:
            return "Both Eyes"
        if self.episode_left_min_openness < self.wink_partner_ratio:
            return "Both Eyes"
        return "Right Eye"

    def update_from_ears(self, left_ear, right_ear, now=None) -> BlinkResult:
        """Process raw EAR values. Useful for tests and the live path."""
        now = time.perf_counter() if now is None else now

        if self.baseline_left is None:
            self.baseline_left = max(left_ear, self.baseline_min)
        if self.baseline_right is None:
            self.baseline_right = max(right_ear, self.baseline_min)

        self.left_closed, self.left_frames, self.left_open_frames = (
            self._update_eye(
                left_ear,
                self.baseline_left,
                self.left_closed,
                self.left_frames,
                self.left_open_frames,
            )
        )
        self.right_closed, self.right_frames, self.right_open_frames = (
            self._update_eye(
                right_ear,
                self.baseline_right,
                self.right_closed,
                self.right_frames,
                self.right_open_frames,
            )
        )

        left_openness = self._openness(left_ear, self.baseline_left)
        right_openness = self._openness(right_ear, self.baseline_right)

        if not self.left_closed:
            self.baseline_left = self._adapt_baseline(
                self.baseline_left, left_ear, False
            )
        if not self.right_closed:
            self.baseline_right = self._adapt_baseline(
                self.baseline_right, right_ear, False
            )

        detected = False

        if self.left_closed or self.right_closed:
            if not self.episode_active:
                self.episode_active = True
                self.episode_left = False
                self.episode_right = False
                self.episode_left_closed_at = None
                self.episode_right_closed_at = None
                self.episode_left_min_openness = left_openness
                self.episode_right_min_openness = right_openness

            self.both_open_since = None
            self.episode_left_min_openness = min(
                self.episode_left_min_openness, left_openness
            )
            self.episode_right_min_openness = min(
                self.episode_right_min_openness, right_openness
            )

            if self.left_closed and not self.episode_left:
                self.episode_left = True
                self.episode_left_closed_at = now
            if self.right_closed and not self.episode_right:
                self.episode_right = True
                self.episode_right_closed_at = now

        elif self.episode_active:
            self.episode_left_min_openness = min(
                self.episode_left_min_openness, left_openness
            )
            self.episode_right_min_openness = min(
                self.episode_right_min_openness, right_openness
            )

            if self.both_open_since is None:
                self.both_open_since = now

            if now - self.both_open_since >= self.settle_s:
                if now >= self.cooldown_until:
                    label = self._classify_episode()
                    if label is not None:
                        self.latest = label
                        detected = True
                        self.cooldown_until = now + self.cooldown_s
                self._clear_episode()

        return BlinkResult(
            latest=self.latest,
            left_closed=self.left_closed,
            right_closed=self.right_closed,
            left_ear=left_ear,
            right_ear=right_ear,
            left_openness=left_openness,
            right_openness=right_openness,
            detected=detected,
        )

    def update(self, landmarks, now=None) -> BlinkResult:
        left_ear = self._ear(landmarks, self.LEFT_EYE)
        right_ear = self._ear(landmarks, self.RIGHT_EYE)
        return self.update_from_ears(left_ear, right_ear, now=now)
