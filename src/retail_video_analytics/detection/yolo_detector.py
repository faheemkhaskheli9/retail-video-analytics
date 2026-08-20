"""Optional YOLO (Ultralytics) detection backend.

This is the backend described in the README/architecture doc for
production-grade accuracy. It is intentionally optional: the ``ultralytics``
package and its model weights are a large download requiring network access,
which is not something this MVP's test suite should depend on. Import is
lazy and guarded so the rest of the package works fine without it installed;
unit tests exercise this module with a stubbed-out model instead of a real
network fetch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Detection, Detector

# COCO class id for "person" in the default Ultralytics models.
_PERSON_CLASS_ID = 0


class YoloDetector(Detector):
    """Wraps an Ultralytics YOLO model behind the shared :class:`Detector` API."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        device: str = "cpu",
        classes: tuple[int, ...] = (_PERSON_CLASS_ID,),
        model: Any | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.classes = classes
        if model is not None:
            # Dependency injection point used by tests (and by callers who
            # already loaded a model) so this class never has to reach the
            # network to be exercised.
            self._model = model
        else:
            self._model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - exercised via mock in tests
            raise RuntimeError(
                "ultralytics is not installed. Install the optional "
                "'yolo' extra (`pip install ultralytics`) or configure "
                "detector.backend: hog to use the offline default."
            ) from exc
        return YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame,
            device=self.device,
            classes=list(self.classes),
            conf=self.confidence_threshold,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", [])
            for box in boxes:
                xyxy = _to_list(box.xyxy)
                conf = _to_scalar(box.conf)
                x1, y1, x2, y2 = xyxy[:4]
                detections.append(
                    Detection(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        score=float(conf),
                        label="person",
                    )
                )
        return detections


def _to_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return list(value)


def _to_scalar(value: Any) -> float:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)
