"""Zone/event engine tests: dwell time, entry/exit counts, cashier absence."""

from __future__ import annotations

from retail_video_analytics.detection.base import Detection
from retail_video_analytics.events.events import EventEngine
from retail_video_analytics.tracking.tracker import Track
from retail_video_analytics.zones.zone import Zone, ZoneManager


def _track_at(track_id: int, cx: float, cy: float, half: float = 2.0) -> Track:
    det = Detection(x1=cx - half, y1=cy - half, x2=cx + half, y2=cy + half, score=1.0)
    return Track(track_id=track_id, detection=det)


def _build_engine(cashier_absence_seconds: float = 3.0) -> EventEngine:
    entrance = Zone(name="entrance", polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)), kind="entrance")
    checkout = Zone(
        name="checkout",
        polygon=((20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0)),
        kind="checkout_counter",
    )
    manager = ZoneManager([entrance, checkout])
    return EventEngine(manager, seconds_per_frame=1.0, cashier_absence_seconds=cashier_absence_seconds)


def test_entry_and_exit_counted_for_entrance_zone():
    engine = _build_engine()

    ev0 = engine.process(0, [_track_at(1, 5, 5)])   # inside entrance
    assert ev0.entries == [("entrance", 1)]

    ev1 = engine.process(1, [_track_at(1, 50, 50)])  # walked out of every zone
    assert ev1.exits == [("entrance", 1)]

    counts = {c.zone_name: c for c in engine.zone_counts()}
    assert counts["entrance"].entries == 1
    assert counts["entrance"].exits == 1


def test_dwell_time_accumulates_while_inside_zone():
    engine = _build_engine()
    for frame_index in range(3):
        engine.process(frame_index, [_track_at(1, 5, 5)])  # stays inside entrance 3 frames

    records = engine.dwell_records()
    assert len(records) == 1
    record = records[0]
    assert record.zone_name == "entrance"
    assert record.track_id == 1
    assert record.seconds == 3.0
    assert record.entries == 1  # only counted once, not once per frame


def test_unique_and_concurrent_customer_counts():
    engine = _build_engine()
    engine.process(0, [_track_at(1, 5, 5)])
    engine.process(1, [_track_at(1, 5, 5), _track_at(2, 6, 6)])
    engine.process(2, [_track_at(2, 6, 6)])

    assert engine.unique_customer_count() == 2
    assert engine.max_concurrent_customers == 2


def test_cashier_absence_alert_fires_after_threshold():
    engine = _build_engine(cashier_absence_seconds=3.0)
    events = [engine.process(i, []) for i in range(4)]  # nobody ever at the counter

    alert_frames = [i for i, ev in enumerate(events) if ev.cashier_absent]
    assert alert_frames == [2]  # 3rd empty frame (1s/frame) crosses the 3s threshold
    assert len(engine.cashier_alerts) == 1
    assert engine.cashier_alerts[0].zone_name == "checkout"


def test_cashier_absence_alert_does_not_refire_once_flagged():
    engine = _build_engine(cashier_absence_seconds=1.0)
    for i in range(5):
        engine.process(i, [])
    assert len(engine.cashier_alerts) == 1  # only one alert, not one per frame past threshold


def test_cashier_presence_clears_absence_streak():
    engine = _build_engine(cashier_absence_seconds=2.0)
    engine.process(0, [])
    engine.process(1, [_track_at(9, 25, 25)])  # cashier shows up at the counter
    events = [engine.process(i, []) for i in range(2, 5)]

    # streak reset when occupied, so it takes cashier_absence_seconds worth of
    # *fresh* empty frames after they leave again before a new alert fires
    alert_frames = [i for i, ev in enumerate(events, start=2) if ev.cashier_absent]
    assert alert_frames == [3]
