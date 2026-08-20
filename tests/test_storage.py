"""SQLite-backed analytics store tests."""

from __future__ import annotations

from pathlib import Path

from retail_video_analytics.events.events import CashierAbsenceAlert, DwellRecord, ZoneCounts
from retail_video_analytics.storage.db import AnalyticsStore


def test_save_and_read_run(tmp_path: Path):
    db_path = tmp_path / "analytics.db"
    with AnalyticsStore(db_path) as store:
        run_id = store.save_run(
            video_source="synthetic",
            frame_count=90,
            unique_customers=3,
            max_concurrent_customers=2,
            zone_counts=[ZoneCounts(zone_name="entrance", entries=3, exits=2)],
            dwell_records=[DwellRecord(track_id=1, zone_name="entrance", seconds=4.5, entries=1)],
            cashier_alerts=[CashierAbsenceAlert(zone_name="checkout", start_frame=10, seconds=5.0)],
        )

    assert run_id >= 1
    assert db_path.exists()

    with AnalyticsStore(db_path) as store:
        run = store.get_run(run_id)
        assert run["video_source"] == "synthetic"
        assert run["frame_count"] == 90
        assert run["unique_customers"] == 3
        assert run["max_concurrent_customers"] == 2

        zone_counts = store.list_zone_counts(run_id)
        assert zone_counts == [{"zone_name": "entrance", "entries": 3, "exits": 2}]


def test_get_run_missing_id_raises_key_error(tmp_path: Path):
    with AnalyticsStore(tmp_path / "analytics.db") as store:
        try:
            store.get_run(999)
            assert False, "expected KeyError"
        except KeyError:
            pass


def test_store_creates_parent_directories(tmp_path: Path):
    nested = tmp_path / "nested" / "dir" / "analytics.db"
    with AnalyticsStore(nested):
        pass
    assert nested.exists()
