"""Configuration loading for the retail video analytics pipeline.

Config is plain YAML so experiments are reproducible and diffable. See
``configs/default.yaml`` for the shape this module expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorConfig:
    backend: str = "hog"  # "hog" (built into OpenCV, no download) or "yolo"
    confidence_threshold: float = 0.4
    yolo_model: str = "yolov8n.pt"
    yolo_device: str = "cpu"


@dataclass
class TrackerConfig:
    max_age: int = 10          # frames a track may go undetected before it is dropped
    iou_threshold: float = 0.3  # min IoU to match a detection to an existing track
    min_hits: int = 1          # detections required before a track is "confirmed"


@dataclass
class ZoneConfig:
    name: str
    polygon: list[list[float]]
    kind: str = "generic"  # "generic", "entrance", "checkout_counter"


@dataclass
class StorageConfig:
    backend: str = "sqlite"
    path: str = "data/analytics.db"
    heatmap_dir: str = "data/heatmaps"


@dataclass
class PipelineConfig:
    video_source: str = "synthetic"
    frame_stride: int = 1
    fps: float = 15.0
    frame_width: int = 640
    frame_height: int = 480
    cashier_absence_seconds: float = 30.0
    synthetic_num_frames: int = 90
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    zones: list[ZoneConfig] = field(default_factory=list)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        data = dict(data)
        detector = DetectorConfig(**data.pop("detector", {}) or {})
        tracker = TrackerConfig(**data.pop("tracker", {}) or {})
        storage = StorageConfig(**data.pop("storage", {}) or {})
        zones_raw = data.pop("zones", []) or []
        zones = [ZoneConfig(**z) for z in zones_raw]
        return cls(
            detector=detector,
            tracker=tracker,
            storage=storage,
            zones=zones,
            **data,
        )


def load_config(path: str | Path) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a YAML file."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    return PipelineConfig.from_dict(data)
