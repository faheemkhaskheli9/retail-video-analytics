"""Config loading tests."""

from __future__ import annotations

from pathlib import Path

from retail_video_analytics.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_default_config():
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert config.video_source == "synthetic"
    assert config.detector.backend == "hog"
    assert config.tracker.max_age == 10
    assert len(config.zones) == 3
    zone_names = {z.name for z in config.zones}
    assert {"entrance", "checkout_counter", "sales_floor"} <= zone_names


def test_load_eval_config():
    config = load_config(REPO_ROOT / "configs" / "eval.yaml")
    assert config.video_source == "synthetic"
    assert config.storage.path == "data/eval.db"


def test_load_config_roundtrips_zone_polygons(tmp_path):
    yaml_text = """
video_source: synthetic
zones:
  - name: a
    kind: entrance
    polygon: [[0, 0], [1, 0], [1, 1], [0, 1]]
"""
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")
    config = load_config(config_path)
    assert config.zones[0].name == "a"
    assert config.zones[0].polygon == [[0, 0], [1, 0], [1, 1], [0, 1]]
