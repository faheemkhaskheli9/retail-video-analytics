"""Tracker tests: ID persistence, aging/dropping, IoU matching."""

from __future__ import annotations

from retail_video_analytics.detection.base import Detection
from retail_video_analytics.tracking.tracker import IOUTracker, iou


def _det(x1, y1, x2, y2, score=0.9):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score)


def test_iou_identical_boxes_is_one():
    box = (0, 0, 10, 10)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    # two 10x10 boxes overlapping in a 5x10 strip -> intersection 50, union 150
    score = iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert abs(score - 50 / 150) < 1e-6


def test_tracker_assigns_stable_id_across_frames():
    tracker = IOUTracker(max_age=5, iou_threshold=0.3)
    tracks1 = tracker.update([_det(0, 0, 20, 40)])
    assert len(tracks1) == 1
    track_id = tracks1[0].track_id

    # small movement frame to frame -> should match the same track id
    tracks2 = tracker.update([_det(3, 2, 23, 42)])
    assert len(tracks2) == 1
    assert tracks2[0].track_id == track_id
    assert tracks2[0].hits == 2


def test_tracker_assigns_new_id_to_new_object():
    tracker = IOUTracker(max_age=5, iou_threshold=0.3)
    tracker.update([_det(0, 0, 20, 40)])
    tracks = tracker.update([_det(0, 0, 20, 40), _det(200, 200, 220, 240)])
    ids = {t.track_id for t in tracks}
    assert len(ids) == 2


def test_tracker_drops_track_after_max_age_missed_frames():
    tracker = IOUTracker(max_age=2, iou_threshold=0.3)
    tracker.update([_det(0, 0, 20, 40)])
    assert len(tracker.tracks) == 1

    tracker.update([])  # miss 1
    assert len(tracker.tracks) == 1
    tracker.update([])  # miss 2
    assert len(tracker.tracks) == 1
    tracker.update([])  # miss 3 -> exceeds max_age, dropped
    assert len(tracker.tracks) == 0


def test_tracker_keeps_track_alive_through_brief_occlusion():
    tracker = IOUTracker(max_age=3, iou_threshold=0.3)
    tracks = tracker.update([_det(0, 0, 20, 40)])
    original_id = tracks[0].track_id

    tracker.update([])  # briefly occluded, one missed frame
    tracks = tracker.update([_det(2, 1, 22, 41)])  # reappears close by

    assert len(tracks) == 1
    assert tracks[0].track_id == original_id
