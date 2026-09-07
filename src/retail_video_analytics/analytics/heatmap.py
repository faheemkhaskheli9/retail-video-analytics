"""Traffic/dwell heatmap rendering from tracked position samples.

The pipeline's :class:`~retail_video_analytics.events.events.EventEngine` records
one ``(x, y)`` centroid per tracked customer per processed frame
(``heatmap_samples``). This module bins those into a 2-D density grid and
renders it as an image, without pulling in matplotlib -- OpenCV (already a
dependency) provides the colormap + PNG encoder.

Outputs per run:
* ``heatmap_<run_id>.png`` -- colour-mapped density image at frame resolution
* ``heatmap_<run_id>.json`` -- the underlying grid + geometry, so the numbers
  are inspectable and re-renderable without re-running the pipeline

Both files are written atomically (temp file + ``os.replace``) so an
interrupted export never leaves a half-written artefact (portfolio rule 1).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

Sample = tuple[float, float]


@dataclass(frozen=True)
class HeatmapGrid:
    counts: np.ndarray  # shape (rows, cols), float; raw sample counts per cell
    cell_size: int
    frame_width: int
    frame_height: int
    sample_count: int

    def to_json_dict(self) -> dict:
        return {
            "cell_size": self.cell_size,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "rows": int(self.counts.shape[0]),
            "cols": int(self.counts.shape[1]),
            "sample_count": self.sample_count,
            "counts": self.counts.astype(float).round(4).tolist(),
        }


@dataclass(frozen=True)
class HeatmapArtifacts:
    png_path: Path
    data_path: Path
    grid: HeatmapGrid


def render_heatmap(
    samples: list[Sample],
    frame_width: int,
    frame_height: int,
    *,
    cell_size: int = 20,
) -> HeatmapGrid:
    """Bin ``samples`` into a density grid.

    Samples outside ``[0, frame_width) x [0, frame_height)`` are clamped to the
    edge rather than dropped -- a detector box can sit slightly outside the
    nominal frame, and silently discarding it would understate an edge hotspot.
    """

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    cols = (frame_width + cell_size - 1) // cell_size
    rows = (frame_height + cell_size - 1) // cell_size
    counts = np.zeros((rows, cols), dtype=np.float64)

    for x, y in samples:
        cx = min(max(int(x // cell_size), 0), cols - 1)
        cy = min(max(int(y // cell_size), 0), rows - 1)
        counts[cy, cx] += 1.0

    return HeatmapGrid(
        counts=counts,
        cell_size=cell_size,
        frame_width=frame_width,
        frame_height=frame_height,
        sample_count=len(samples),
    )


def _colormap_image(grid: HeatmapGrid, *, blur_sigma: float = 0.0) -> np.ndarray:
    counts = grid.counts
    if blur_sigma > 0:
        counts = cv2.GaussianBlur(counts, ksize=(0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    peak = float(counts.max())
    if peak <= 0:
        normed = np.zeros_like(counts, dtype=np.uint8)
    else:
        normed = np.clip(counts / peak * 255.0, 0, 255).astype(np.uint8)

    upscaled = cv2.resize(
        normed,
        (grid.frame_width, grid.frame_height),
        interpolation=cv2.INTER_NEAREST if blur_sigma == 0 else cv2.INTER_LINEAR,
    )
    return cv2.applyColorMap(upscaled, cv2.COLORMAP_JET)


def _atomic_write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def save_heatmap(
    grid: HeatmapGrid,
    out_dir: str | Path,
    run_id: int,
    *,
    blur_sigma: float = 1.0,
) -> HeatmapArtifacts:
    out_dir = Path(out_dir)
    png_path = out_dir / f"heatmap_{run_id}.png"
    data_path = out_dir / f"heatmap_{run_id}.json"

    image = _colormap_image(grid, blur_sigma=blur_sigma)
    ok, buf = cv2.imencode(".png", image)
    if not ok:  # pragma: no cover - cv2 PNG encode does not realistically fail here
        raise RuntimeError("failed to PNG-encode heatmap image")
    _atomic_write(png_path, buf.tobytes())
    _atomic_write(
        data_path,
        json.dumps(grid.to_json_dict(), indent=2).encode("utf-8"),
    )
    return HeatmapArtifacts(png_path=png_path, data_path=data_path, grid=grid)
