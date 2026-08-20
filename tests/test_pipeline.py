"""End-to-end pipeline tests using the synthetic scene (no real video/GPU needed)."""

from __future__ import annotations

from pathlib import Path

from retail_video_analytics.config import load_config
from retail_video_analytics.pipeline import VideoAnalyticsPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_config(tmp_path: Path, num_frames: int = 60):
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    config.video_source = "synthetic"
    config.synthetic_num_frames = num_frames
    config.storage.path = str(tmp_path / "analytics.db")
    return config


def test_pipeline_runs_end_to_end_on_synthetic_scene(tmp_path):
    config = _synthetic_config(tmp_path)
    pipeline = VideoAnalyticsPipeline(config)
    summary = pipeline.run(persist=True)

    assert summary.frame_count == 60
    # 3 synthetic actors walk through the scene -> the tracker should find them all.
    assert summary.unique_customers == 3
    assert summary.max_concurrent_customers >= 1
    assert summary.run_id is not None


def test_pipeline_counts_entrance_traffic():
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    config.video_source = "synthetic"
    config.synthetic_num_frames = 90
    pipeline = VideoAnalyticsPipeline(config)
    summary = pipeline.run(persist=False)

    entrance = next(z for z in summary.zone_counts if z.zone_name == "entrance")
    # All three synthetic actors start inside the entrance zone.
    assert entrance.entries >= 1


def test_pipeline_records_dwell_time_for_loitering_actor():
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    config.video_source = "synthetic"
    config.synthetic_num_frames = 90
    pipeline = VideoAnalyticsPipeline(config)
    summary = pipeline.run(persist=False)

    assert len(summary.dwell_records) > 0
    assert all(record.seconds >= 0 for record in summary.dwell_records)


def test_pipeline_flags_cashier_absence_when_counter_never_staffed():
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    config.video_source = "synthetic"
    config.synthetic_num_frames = 90
    config.cashier_absence_seconds = 1.0  # low threshold so the synthetic run trips it
    pipeline = VideoAnalyticsPipeline(config)
    summary = pipeline.run(persist=False)

    # The checkout counter sits empty for the first several dozen frames
    # while actors are still walking in from the entrance, so at a 1s
    # threshold an absence alert should fire at least once.
    assert len(summary.cashier_alerts) >= 1


def test_pipeline_summary_serializes_to_dict():
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    config.video_source = "synthetic"
    config.synthetic_num_frames = 30
    pipeline = VideoAnalyticsPipeline(config)
    summary = pipeline.run(persist=False)

    payload = summary.as_dict()
    assert payload["frame_count"] == 30
    assert isinstance(payload["zone_counts"], list)
    assert isinstance(payload["dwell_records"], list)
