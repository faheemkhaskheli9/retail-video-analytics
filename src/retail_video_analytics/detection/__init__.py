"""Detection backends and factory."""

from __future__ import annotations

from ..config import DetectorConfig
from .base import Detection, Detector
from .hog_detector import HOGPersonDetector

__all__ = [
    "Detection",
    "Detector",
    "HOGPersonDetector",
    "build_detector",
]


def build_detector(config: DetectorConfig) -> Detector:
    """Construct the configured detector backend."""
    if config.backend == "hog":
        return HOGPersonDetector(confidence_threshold=config.confidence_threshold)
    if config.backend == "yolo":
        from .yolo_detector import YoloDetector

        return YoloDetector(
            model_path=config.yolo_model,
            confidence_threshold=config.confidence_threshold,
            device=config.yolo_device,
        )
    raise ValueError(f"Unknown detector backend: {config.backend!r}")
