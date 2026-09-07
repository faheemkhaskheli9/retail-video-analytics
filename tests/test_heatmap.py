"""Heatmap rendering + persistence tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from retail_video_analytics.analytics.heatmap import render_heatmap, save_heatmap
from retail_video_analytics.config import load_config
from retail_video_analytics.heatmap_cli import main as heatmap_main
from retail_video_analytics.pipeline import VideoAnalyticsPipeline
from retail_video_analytics.storage.db import AnalyticsStore

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_render_preserves_total_sample_count():
    samples = [(10.0, 10.0), (10.0, 12.0), (600.0, 400.0), (0.0, 0.0)]
    grid = render_heatmap(samples, 640, 480, cell_size=20)
    assert grid.counts.sum() == len(samples)
    assert grid.counts.shape == (24, 32)


def test_render_clamps_out_of_frame_samples():
    grid = render_heatmap([(-5.0, -5.0), (10_000.0, 10_000.0)], 100, 100, cell_size=25)
    assert grid.counts.sum() == 2
    assert grid.counts[0, 0] == 1  # clamped to top-left
    assert grid.counts[-1, -1] == 1  # clamped to bottom-right


def test_render_hotspot_is_the_argmax_cell():
    samples = [(300.0, 200.0)] * 50 + [(10.0, 10.0)] * 3
    grid = render_heatmap(samples, 640, 480, cell_size=20)
    cy, cx = np.unravel_index(int(grid.counts.argmax()), grid.counts.shape)
    assert (cx, cy) == (300 // 20, 200 // 20)


def test_render_rejects_bad_geometry():
    with pytest.raises(ValueError):
        render_heatmap([], 0, 100)
    with pytest.raises(ValueError):
        render_heatmap([], 100, 100, cell_size=0)


def test_save_writes_png_and_json(tmp_path):
    grid = render_heatmap([(100.0, 100.0)] * 5, 320, 240, cell_size=40)
    artifacts = save_heatmap(grid, tmp_path, run_id=7)

    assert artifacts.png_path == tmp_path / "heatmap_7.png"
    assert artifacts.data_path == tmp_path / "heatmap_7.json"

    img = cv2.imread(str(artifacts.png_path))
    assert img is not None and img.shape == (240, 320, 3)

    data = json.loads(artifacts.data_path.read_text())
    assert data["sample_count"] == 5
    assert data["rows"] == 6 and data["cols"] == 8
    assert sum(sum(row) for row in data["counts"]) == 5

    # no leftover temp files
    assert not list(tmp_path.glob("*.tmp"))


def _synthetic_config(tmp_path):
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    config.video_source = "synthetic"
    config.synthetic_num_frames = 40
    config.storage.path = str(tmp_path / "analytics.db")
    config.storage.heatmap_dir = str(tmp_path / "heatmaps")
    return config


def test_pipeline_persists_heatmap_per_run(tmp_path):
    summary = VideoAnalyticsPipeline(_synthetic_config(tmp_path)).run(persist=True)
    assert summary.heatmap_png and Path(summary.heatmap_png).is_file()
    assert summary.heatmap_data and Path(summary.heatmap_data).is_file()

    with AnalyticsStore(tmp_path / "analytics.db") as store:
        samples = store.list_position_samples(summary.run_id)
    assert len(samples) > 0


def test_heatmap_cli_regenerates_from_stored_run(tmp_path, capsys):
    summary = VideoAnalyticsPipeline(_synthetic_config(tmp_path)).run(persist=True)
    out_dir = tmp_path / "regen"

    rc = heatmap_main(
        [
            "--db",
            str(tmp_path / "analytics.db"),
            "--run-id",
            str(summary.run_id),
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / f"heatmap_{summary.run_id}.png").is_file()
    assert (out_dir / f"heatmap_{summary.run_id}.json").is_file()


def test_heatmap_cli_errors_on_unknown_run(tmp_path):
    with AnalyticsStore(tmp_path / "analytics.db"):
        pass
    rc = heatmap_main(["--db", str(tmp_path / "analytics.db"), "--run-id", "999"])
    assert rc == 1


def test_heatmap_cli_errors_when_run_has_no_samples(tmp_path):
    from retail_video_analytics.events.events import ZoneCounts

    with AnalyticsStore(tmp_path / "analytics.db") as store:
        run_id = store.save_run(
            video_source="synthetic",
            frame_count=1,
            unique_customers=0,
            max_concurrent_customers=0,
            zone_counts=[ZoneCounts(zone_name="z")],
            dwell_records=[],
            cashier_alerts=[],
        )
    rc = heatmap_main(["--db", str(tmp_path / "analytics.db"), "--run-id", str(run_id)])
    assert rc == 1
