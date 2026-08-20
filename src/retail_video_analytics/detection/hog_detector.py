"""Classical CPU person detector built on OpenCV's shipped HOG + SVM model.

This backend requires no model download and no GPU -- the descriptor and its
pre-trained linear SVM weights ship inside opencv-python itself -- which
makes it the default for this portfolio MVP so the pipeline is runnable
offline and in CI. ``YoloDetector`` (see ``yolo_detector.py``) implements the
same interface for anyone who wants higher accuracy and has network access to
fetch weights.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import Detection, Detector


class HOGPersonDetector(Detector):
    """Person detector using ``cv2.HOGDescriptor``'s default people detector."""

    def __init__(
        self,
        confidence_threshold: float = 0.4,
        win_stride: tuple[int, int] = (8, 8),
        scale: float = 1.05,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.win_stride = win_stride
        self.scale = scale
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []
        rects, weights = self._hog.detectMultiScale(
            frame,
            winStride=self.win_stride,
            scale=self.scale,
        )
        detections: list[Detection] = []
        for (x, y, w, h), weight in zip(rects, weights):
            score = float(weight)
            # HOG's SVM decision function is unbounded; squash to (0, 1) so
            # it behaves like every other backend's confidence score.
            confidence = 1.0 / (1.0 + np.exp(-score))
            if confidence < self.confidence_threshold:
                continue
            detections.append(
                Detection(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                    score=confidence,
                    label="person",
                )
            )
        return detections
