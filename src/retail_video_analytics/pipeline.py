"""End-to-end orchestration: detection -> tracking -> zone/event logic -> storage.

Mirrors the pipeline diagram in ``docs/architecture.md``:

    Camera Feed -> Detection -> Tracking -> Zone/Event Logic -> Database

``video_source`` in the config selects the input: a path to a real video file
(read with OpenCV) or the literal string ``"synthetic"``, which drives the
offline demo/test scene from :mod:`retail_video_analytics.synthetic`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from .analytics.heatmap import render_heatmap, save_heatmap
from .config import PipelineConfig
from .detection.base import Detection, Detector
from .detection import build_detector
from .events.events import CashierAbsenceAlert, DwellRecord, EventEngine, ZoneCounts
from .storage.db import AnalyticsStore
from .synthetic import SimulatedDetector, default_actor_paths, generate_synthetic_detections
from .tracking.tracker import IOUTracker
from .zones.zone import ZoneManager


@dataclass
class RunSummary:
    frame_count: int
    unique_customers: int
    max_concurrent_customers: int
    zone_counts: list[ZoneCounts]
    dwell_records: list[DwellRecord]
    cashier_alerts: list[CashierAbsenceAlert]
    run_id: int | None = None
    heatmap_png: str | None = None
    heatmap_data: str | None = None

    def as_dict(self) -> dict:
        return {
            "frame_count": self.frame_count,
            "unique_customers": self.unique_customers,
            "max_concurrent_customers": self.max_concurrent_customers,
            "zone_counts": [
                {"zone_name": z.zone_name, "entries": z.entries, "exits": z.exits}
                for z in self.zone_counts
            ],
            "dwell_records": [
                {
                    "track_id": d.track_id,
                    "zone_name": d.zone_name,
                    "seconds": round(d.seconds, 2),
                    "entries": d.entries,
                }
                for d in self.dwell_records
            ],
            "cashier_alerts": [
                {
                    "zone_name": a.zone_name,
                    "start_frame": a.start_frame,
                    "seconds": round(a.seconds, 2),
                }
                for a in self.cashier_alerts
            ],
            "run_id": self.run_id,
            "heatmap_png": self.heatmap_png,
            "heatmap_data": self.heatmap_data,
        }


def _iter_video_frames(path: str, frame_stride: int = 1) -> Iterator[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {path}")
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % frame_stride == 0:
                yield frame
            index += 1
    finally:
        cap.release()


class VideoAnalyticsPipeline:
    """Wires detection, tracking, zone/event logic, and storage together."""

    def __init__(self, config: PipelineConfig, detector: Detector | None = None) -> None:
        self.config = config
        self.zone_manager = ZoneManager.from_config(config.zones)
        self.tracker = IOUTracker(
            max_age=config.tracker.max_age,
            iou_threshold=config.tracker.iou_threshold,
            min_hits=config.tracker.min_hits,
        )
        self.event_engine = EventEngine(
            zone_manager=self.zone_manager,
            seconds_per_frame=(config.frame_stride / config.fps) if config.fps else 0.0,
            cashier_absence_seconds=config.cashier_absence_seconds,
        )
        self._detector_override = detector

    def _resolve_detector_and_frames(self) -> tuple[Detector, Iterable[np.ndarray | None]]:
        if self._detector_override is not None:
            frames = self._synthetic_or_video_frames()
            return self._detector_override, frames

        if self.config.video_source == "synthetic":
            num_frames = self.config.synthetic_num_frames
            paths = default_actor_paths(640, 480, num_frames=num_frames)
            frame_detections = generate_synthetic_detections(paths, num_frames=num_frames)
            detector = SimulatedDetector(frame_detections)
            frames: Iterable[np.ndarray | None] = (None for _ in range(len(frame_detections)))
            return detector, frames

        detector = build_detector(self.config.detector)
        frames = _iter_video_frames(self.config.video_source, self.config.frame_stride)
        return detector, frames

    def _synthetic_or_video_frames(self) -> Iterable[np.ndarray | None]:
        if self.config.video_source == "synthetic":
            return (None for _ in range(self.config.synthetic_num_frames))
        return _iter_video_frames(self.config.video_source, self.config.frame_stride)

    def run(self, persist: bool = True) -> RunSummary:
        detector, frames = self._resolve_detector_and_frames()

        frame_count = 0
        for frame_index, frame in enumerate(frames):
            detections: list[Detection] = detector.detect(frame)
            tracks = self.tracker.update(detections)
            self.event_engine.process(frame_index, tracks)
            frame_count += 1

        summary = RunSummary(
            frame_count=frame_count,
            unique_customers=self.event_engine.unique_customer_count(),
            max_concurrent_customers=self.event_engine.max_concurrent_customers,
            zone_counts=self.event_engine.zone_counts(),
            dwell_records=self.event_engine.dwell_records(),
            cashier_alerts=self.event_engine.cashier_alerts,
        )

        if persist:
            with AnalyticsStore(self.config.storage.path) as store:
                run_id = store.save_run(
                    video_source=self.config.video_source,
                    frame_count=summary.frame_count,
                    unique_customers=summary.unique_customers,
                    max_concurrent_customers=summary.max_concurrent_customers,
                    zone_counts=summary.zone_counts,
                    dwell_records=summary.dwell_records,
                    cashier_alerts=summary.cashier_alerts,
                    position_samples=self.event_engine.heatmap_samples,
                    frame_width=self.config.frame_width,
                    frame_height=self.config.frame_height,
                )
                summary.run_id = run_id

            grid = render_heatmap(
                self.event_engine.heatmap_samples,
                self.config.frame_width,
                self.config.frame_height,
            )
            artifacts = save_heatmap(grid, self.config.storage.heatmap_dir, run_id)
            summary.heatmap_png = str(artifacts.png_path)
            summary.heatmap_data = str(artifacts.data_path)

        return summary
