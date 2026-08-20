"""Synthetic scene generator tests."""

from __future__ import annotations

import numpy as np

from retail_video_analytics.synthetic import (
    SimulatedDetector,
    default_actor_paths,
    generate_synthetic_detections,
    generate_synthetic_frames,
)


def test_default_actor_paths_produces_three_actors():
    paths = default_actor_paths(640, 480, num_frames=90)
    assert len(paths) == 3


def test_generate_synthetic_detections_matches_frame_count():
    paths = default_actor_paths(640, 480, num_frames=20)
    frames = generate_synthetic_detections(paths, num_frames=20)
    assert len(frames) == 20
    assert all(isinstance(f, list) for f in frames)


def test_generate_synthetic_frames_yields_correct_shape():
    paths = default_actor_paths(320, 240, num_frames=5)
    frames = list(generate_synthetic_frames(paths, num_frames=5, width=320, height=240))
    assert len(frames) == 5
    for frame in frames:
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (240, 320, 3)


def test_simulated_detector_replays_ground_truth_without_noise():
    paths = default_actor_paths(640, 480, num_frames=10)
    ground_truth = generate_synthetic_detections(paths, num_frames=10)
    detector = SimulatedDetector(ground_truth)

    for frame_index in range(10):
        detections = detector.detect(None)
        assert detections == ground_truth[frame_index]


def test_simulated_detector_miss_rate_drops_detections():
    # 5 detections in a single synthetic frame.
    from retail_video_analytics.detection.base import Detection

    ground_truth = [[Detection(0, 0, 10, 10, 1.0) for _ in range(5)]]
    detector = SimulatedDetector(ground_truth, miss_rate=1.0, seed=1)
    assert detector.detect(None) == []


def test_simulated_detector_resets_cursor():
    from retail_video_analytics.detection.base import Detection

    ground_truth = [[Detection(0, 0, 10, 10, 1.0)], []]
    detector = SimulatedDetector(ground_truth)
    assert len(detector.detect(None)) == 1
    assert len(detector.detect(None)) == 0
    assert detector.detect(None) == []  # past the end
    detector.reset()
    assert len(detector.detect(None)) == 1
