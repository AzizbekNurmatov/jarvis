"""Head angles and lightweight exponential smoothing."""
from __future__ import annotations

import cv2
import numpy as np


def euler_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Return [yaw, pitch, roll] in degrees from a 3x3 rotation matrix.

    OpenCV RQDecomp3x3 returns Euler rotations around X, Y, Z in degrees.
    We label those pitch, yaw, roll. Values are estimates, not gaze direction.
    """
    angles, *_ = cv2.RQDecomp3x3(np.asarray(rotation, dtype=np.float64))
    pitch, yaw, roll = angles
    return np.array([yaw, pitch, roll], dtype=np.float64)


class PoseSmoother:
    def __init__(self, alpha: float = 0.28):
        self.alpha = alpha
        self.previous: np.ndarray | None = None

    def update(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self.previous is None:
            self.previous = value.copy()
        else:
            # Wrap differences so yaw/roll crossings around +/-180 do not jump.
            diff = (value - self.previous + 180.0) % 360.0 - 180.0
            self.previous = self.previous + self.alpha * diff
        return self.previous.copy()

    def reset(self) -> None:
        self.previous = None
