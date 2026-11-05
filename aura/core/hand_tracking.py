"""MediaPipe Hands landmark tracking and overlay drawing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


# BGR colors matched to AURA's cyan / purple accents.
LEFT_DOT = (210, 200, 80)      # soft cyan-yellow
LEFT_BONE = (220, 180, 60)
RIGHT_DOT = (220, 140, 210)    # soft purple
RIGHT_BONE = (200, 110, 190)
LABEL_BG = (20, 18, 12)


@dataclass
class TrackedHand:
    label: str  # anatomical "Left" or "Right"
    landmarks: Sequence  # MediaPipe landmark list (21)
    score: float


class HandTracker:
    """
    Lazy MediaPipe Hands wrapper.

    Create at most one Hands graph. Enabling/disabling closes and
    rebuilds cleanly so toggles do not leak resources.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.5,
        model_complexity: int = 0,
        mirrored: bool = True,
    ):
        self.max_num_hands = max_num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.model_complexity = model_complexity
        # Input frames are already selfie-flipped in main.py.
        self.mirrored = mirrored

        self._hands: Optional[mp.solutions.hands.Hands] = None
        self.enabled = False
        self._connections = mp.solutions.hands.HAND_CONNECTIONS

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self.enabled and (
            (enabled and self._hands is not None)
            or (not enabled and self._hands is None)
        ):
            return

        if enabled:
            self.close()
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence,
            )
            self.enabled = True
        else:
            self.close()

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None
        self.enabled = False

    def _anatomical_label(self, handedness_label: str) -> str:
        """
        MediaPipe handedness is relative to the input image.

        After a horizontal selfie flip, the classifier's Left/Right is
        mirrored relative to the person's body, so we invert labels.
        """
        label = handedness_label.strip().title()
        if not self.mirrored:
            return label if label in ("Left", "Right") else "Right"
        if label == "Left":
            return "Right"
        if label == "Right":
            return "Left"
        return "Right"

    def process(self, rgb_image: np.ndarray) -> List[TrackedHand]:
        """Run inference on an RGB frame. Returns [] when disabled."""
        if not self.enabled or self._hands is None:
            return []

        result = self._hands.process(rgb_image)
        if not result.multi_hand_landmarks:
            return []

        hands: List[TrackedHand] = []
        handedness_list = result.multi_handedness or []

        for index, landmarks in enumerate(result.multi_hand_landmarks):
            score = 0.0
            raw_label = "Right"
            if index < len(handedness_list):
                category = handedness_list[index].classification[0]
                raw_label = category.label
                score = float(category.score)

            hands.append(
                TrackedHand(
                    label=self._anatomical_label(raw_label),
                    landmarks=landmarks.landmark,
                    score=score,
                )
            )

        return hands

    def draw(
        self,
        frame_bgr: np.ndarray,
        hands: List[TrackedHand],
    ) -> None:
        """Draw landmarks, bones, and LEFT/RIGHT labels onto a BGR frame."""
        if not hands:
            return

        height, width = frame_bgr.shape[:2]

        for hand in hands:
            is_left = hand.label == "Left"
            dot = LEFT_DOT if is_left else RIGHT_DOT
            bone = LEFT_BONE if is_left else RIGHT_BONE

            points: List[Tuple[int, int]] = []
            for landmark in hand.landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))
                cv2.circle(frame_bgr, (x, y), 3, dot, -1, cv2.LINE_AA)

            for start_idx, end_idx in self._connections:
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(
                        frame_bgr,
                        points[start_idx],
                        points[end_idx],
                        bone,
                        2,
                        cv2.LINE_AA,
                    )

            # Wrist is landmark 0 — good anchor for the label.
            wx, wy = points[0]
            text = hand.label.upper()
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            pad = 4
            x1 = max(0, wx - pad)
            y1 = max(0, wy - th - pad * 3)
            x2 = min(width - 1, x1 + tw + pad * 2)
            y2 = min(height - 1, y1 + th + pad * 2)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), LABEL_BG, -1)
            cv2.putText(
                frame_bgr,
                text,
                (x1 + pad, y2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                dot,
                1,
                cv2.LINE_AA,
            )
