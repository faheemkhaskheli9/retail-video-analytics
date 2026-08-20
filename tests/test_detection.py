"""Detection backend tests."""

from __future__ import annotations

import numpy as np
import pytest

from retail_video_analytics.config import DetectorConfig
from retail_video_analytics.detection import build_detector
from retail_video_analytics.detection.base import Detection
from retail_video_analytics.detection.hog_detector import HOGPersonDetector


def test_detection_geometry_helpers():
    det = Detection(x1=10, y1=20, x2=30, y2=60, score=0.9, label="person")
    assert det.bbox == (10, 20, 30, 60)
    assert det.centroid == (20.0, 40.0)
    assert det.area == 20 * 40


def test_hog_detector_runs_without_crashing_on_blank_frame():
    # A real HOG person detector will not "find" anyone in a blank/noise
    # frame; this test only asserts the real OpenCV pipeline runs end to end
    # (loads the built-in SVM, executes detectMultiScale) and returns a
    # well-formed (possibly empty) detection list.
    detector = HOGPersonDetector(confidence_threshold=0.4)
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    detections = detector.detect(frame)
    assert isinstance(detections, list)
    for det in detections:
        assert isinstance(det, Detection)
        assert 0.0 <= det.score <= 1.0


def test_hog_detector_handles_empty_frame():
    detector = HOGPersonDetector()
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_build_detector_hog_backend():
    detector = build_detector(DetectorConfig(backend="hog"))
    assert isinstance(detector, HOGPersonDetector)


def test_build_detector_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_detector(DetectorConfig(backend="not-a-backend"))


def test_yolo_detector_uses_injected_model_without_network():
    """YOLO backend is exercised via dependency injection so the test suite
    never needs network access or a real model download."""
    from retail_video_analytics.detection.yolo_detector import YoloDetector

    class _FakeBox:
        def __init__(self, xyxy, conf):
            self.xyxy = [xyxy]
            self.conf = [conf]

    class _FakeResult:
        def __init__(self, boxes):
            self.boxes = boxes

    class _FakeModel:
        def predict(self, frame, device, classes, conf, verbose):
            return [_FakeResult([_FakeBox([10, 10, 50, 90], 0.87)])]

    detector = YoloDetector(model=_FakeModel())
    detections = detector.detect(frame=np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].bbox == (10.0, 10.0, 50.0, 90.0)
    assert detections[0].score == pytest.approx(0.87)


def test_build_detector_yolo_without_ultralytics_raises_clear_error():
    """When ultralytics isn't installed, the failure should be a clear
    RuntimeError explaining how to fix it, not an opaque ImportError."""
    detector_config = DetectorConfig(backend="yolo", yolo_model="does-not-matter.pt")
    try:
        import ultralytics  # noqa: F401

        pytest.skip("ultralytics is installed in this environment; nothing to assert")
    except ImportError:
        with pytest.raises(RuntimeError):
            build_detector(detector_config)
