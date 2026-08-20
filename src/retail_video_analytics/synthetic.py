"""Synthetic scene generator used for the offline demo and for tests.

No public retail CCTV dataset ships with ready-to-use ground truth for
entry/exit and dwell-time events, and recording real store footage is out of
scope for a portfolio project. Instead this module generates a small
synthetic scene: a handful of "customer" actors walking straight-line paths
across a frame, with known ground-truth positions every frame.

Two things come out of it:

* :func:`generate_synthetic_frames` -- rendered BGR frames (filled rectangles
  on a blank floor) purely so the pipeline has something to optionally write
  out as a demo video.
* :class:`SimulatedDetector` -- a :class:`~retail_video_analytics.detection.base.Detector`
  that replays the ground-truth boxes with configurable noise and a missed-
  detection rate, standing in for a real camera + detector when no camera
  feed is available. It is explicitly a simulation, not a claim that HOG/YOLO
  were run on synthetic footage -- see ``docs/architecture.md``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .detection.base import Detection, Detector


@dataclass(frozen=True)
class ActorPath:
    start: tuple[float, float]
    end: tuple[float, float]
    width: float = 40.0
    height: float = 90.0
    start_frame: int = 0
    end_frame: int = 100

    def position_at(self, frame_index: int) -> tuple[float, float] | None:
        if frame_index < self.start_frame or frame_index > self.end_frame:
            return None
        span = max(1, self.end_frame - self.start_frame)
        t = (frame_index - self.start_frame) / span
        x = self.start[0] + (self.end[0] - self.start[0]) * t
        y = self.start[1] + (self.end[1] - self.start[1]) * t
        return (x, y)

    def bbox_at(self, frame_index: int) -> tuple[float, float, float, float] | None:
        pos = self.position_at(frame_index)
        if pos is None:
            return None
        cx, cy = pos
        return (cx - self.width / 2, cy - self.height / 2, cx + self.width / 2, cy + self.height / 2)


def default_actor_paths(frame_width: int, frame_height: int, num_frames: int) -> list[ActorPath]:
    """A few plausible customer walks: through an entrance zone to a checkout counter."""
    entrance_x = frame_width * 0.1
    checkout_x = frame_width * 0.85
    mid_y = frame_height * 0.5
    return [
        ActorPath(
            start=(entrance_x, mid_y - 60),
            end=(checkout_x, mid_y - 60),
            start_frame=0,
            end_frame=int(num_frames * 0.6),
        ),
        ActorPath(
            start=(entrance_x, mid_y + 20),
            end=(checkout_x, mid_y + 20),
            start_frame=int(num_frames * 0.1),
            end_frame=int(num_frames * 0.8),
        ),
        ActorPath(
            start=(entrance_x, mid_y + 90),
            end=(entrance_x, mid_y + 90),  # loiters near the entrance the whole time
            start_frame=int(num_frames * 0.2),
            end_frame=int(num_frames * 0.9),
        ),
    ]


def generate_synthetic_detections(
    paths: list[ActorPath], num_frames: int
) -> list[list[Detection]]:
    """Ground-truth per-frame detections implied by ``paths`` (no noise)."""
    frames: list[list[Detection]] = []
    for frame_index in range(num_frames):
        dets: list[Detection] = []
        for path in paths:
            bbox = path.bbox_at(frame_index)
            if bbox is not None:
                dets.append(Detection(*bbox, score=1.0, label="person"))
        frames.append(dets)
    return frames


def generate_synthetic_frames(
    paths: list[ActorPath],
    num_frames: int,
    width: int = 640,
    height: int = 480,
):
    """Yield rendered BGR frames (numpy arrays) for the given actor paths."""
    import cv2

    for frame_index in range(num_frames):
        frame = np.full((height, width, 3), 230, dtype=np.uint8)
        for path in paths:
            bbox = path.bbox_at(frame_index)
            if bbox is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 90, 200), thickness=-1)
        yield frame


class SimulatedDetector(Detector):
    """Replays a precomputed ground-truth detection sequence with optional noise.

    Used only for the offline synthetic demo/tests where no real camera feed
    is available. Real deployments use :class:`HOGPersonDetector` or
    :class:`YoloDetector` against actual footage instead.
    """

    def __init__(
        self,
        frame_detections: list[list[Detection]],
        noise_std: float = 0.0,
        miss_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        self._frame_detections = frame_detections
        self.noise_std = noise_std
        self.miss_rate = miss_rate
        self._rng = random.Random(seed)
        self._cursor = 0

    def detect(self, frame: np.ndarray | None = None) -> list[Detection]:
        if self._cursor >= len(self._frame_detections):
            return []
        dets = self._frame_detections[self._cursor]
        self._cursor += 1

        result: list[Detection] = []
        for det in dets:
            if self.miss_rate and self._rng.random() < self.miss_rate:
                continue
            if self.noise_std:
                dx = self._rng.gauss(0, self.noise_std)
                dy = self._rng.gauss(0, self.noise_std)
                det = Detection(
                    x1=det.x1 + dx,
                    y1=det.y1 + dy,
                    x2=det.x2 + dx,
                    y2=det.y2 + dy,
                    score=det.score,
                    label=det.label,
                )
            result.append(det)
        return result

    def reset(self) -> None:
        self._cursor = 0
