
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class BlinkResult:
    latest: str = "None"
    left_closed: bool = False
    right_closed: bool = False
    left_ear: float = 0.0
    right_ear: float = 0.0
    detected: bool = False


class BlinkDetector:
    """
    MediaPipe Face Mesh blink detection.

    Labels refer to the person's anatomical left/right,
    not the left/right side of the displayed camera image.

    Only the latest result is retained; no blink history.
    """

    # Six landmarks around each eye.
    LEFT_EYE = (362, 385, 387, 263, 373, 380)
    RIGHT_EYE = (33, 160, 158, 133, 153, 144)

    def __init__(
        self,
        close_ratio=0.72,
        open_ratio=0.84,
        min_closed_frames=2,
    ):
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.min_closed_frames = min_closed_frames

        self.baseline_left = None
        self.baseline_right = None

        self.latest = "None"

        self.left_closed = False
        self.right_closed = False

        self.left_frames = 0
        self.right_frames = 0

        self.in_blink = False
        self.episode_left = False
        self.episode_right = False
        self.open_frames = 0

    @staticmethod
    def _distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _ear(self, landmarks, indices):
        p = [landmarks[i] for i in indices]

        vertical_1 = self._distance(p[1], p[5])
        vertical_2 = self._distance(p[2], p[4])
        horizontal = self._distance(p[0], p[3])

        if horizontal < 1e-6:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def reset(self):
        self.baseline_left = None
        self.baseline_right = None

        self.left_closed = False
        self.right_closed = False

        self.left_frames = 0
        self.right_frames = 0

        self.in_blink = False
        self.episode_left = False
        self.episode_right = False
        self.open_frames = 0

        self.latest = "None"

    def face_lost(self):
        # Keep the latest blink, but discard an incomplete event.
        self.in_blink = False
        self.episode_left = False
        self.episode_right = False
        self.left_closed = False
        self.right_closed = False
        self.left_frames = 0
        self.right_frames = 0
        self.open_frames = 0

    def _update_eye(self, ear, baseline, closed, frames):
        close_threshold = baseline * self.close_ratio
        open_threshold = baseline * self.open_ratio

        if closed:
            if ear >= open_threshold:
                closed = False
                frames = 0
        else:
            if ear < close_threshold:
                frames += 1
                if frames >= self.min_closed_frames:
                    closed = True
            else:
                frames = 0

        return closed, frames

    def update(self, landmarks) -> BlinkResult:
        left_ear = self._ear(landmarks, self.LEFT_EYE)
        right_ear = self._ear(landmarks, self.RIGHT_EYE)

        if self.baseline_left is None:
            self.baseline_left = left_ear
        if self.baseline_right is None:
            self.baseline_right = right_ear

        self.left_closed, self.left_frames = self._update_eye(
            left_ear,
            self.baseline_left,
            self.left_closed,
            self.left_frames,
        )

        self.right_closed, self.right_frames = self._update_eye(
            right_ear,
            self.baseline_right,
            self.right_closed,
            self.right_frames,
        )

        # Slowly learn the user's normal open-eye EAR.
        if not self.left_closed and not self.in_blink:
            if left_ear > self.baseline_left * 0.84:
                self.baseline_left = (
                    0.995 * self.baseline_left + 0.005 * left_ear
                )

        if not self.right_closed and not self.in_blink:
            if right_ear > self.baseline_right * 0.84:
                self.baseline_right = (
                    0.995 * self.baseline_right + 0.005 * right_ear
                )

        detected = False

        if self.left_closed or self.right_closed:
            self.in_blink = True
            self.open_frames = 0

            self.episode_left |= self.left_closed
            self.episode_right |= self.right_closed

        elif self.in_blink:
            self.open_frames += 1

            # Confirm reopening before finishing the event.
            if self.open_frames >= 2:
                if self.episode_left and self.episode_right:
                    self.latest = "Both Eyes"
                elif self.episode_left:
                    self.latest = "Left Eye"
                elif self.episode_right:
                    self.latest = "Right Eye"

                detected = True
                self.in_blink = False
                self.episode_left = False
                self.episode_right = False
                self.open_frames = 0

        return BlinkResult(
            latest=self.latest,
            left_closed=self.left_closed,
            right_closed=self.right_closed,
            left_ear=left_ear,
            right_ear=right_ear,
            detected=detected,
        )
