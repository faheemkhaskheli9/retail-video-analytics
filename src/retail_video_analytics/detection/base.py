"""Detector interface shared by every detection backend.

Keeping detection behind a small interface means the tracking, zone, and
event logic never care whether boxes came from a classical CPU detector or a
YOLO model -- swapping backends is a one-line config change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detected object in one frame, in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str = "person"

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class Detector(ABC):
    """Abstract base class for a per-frame object detector."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for a single BGR frame (as produced by OpenCV)."""
        raise NotImplementedError
