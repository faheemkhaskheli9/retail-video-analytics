"""Zone polygon membership tests."""

from __future__ import annotations

from retail_video_analytics.config import ZoneConfig
from retail_video_analytics.zones.zone import Zone, ZoneManager


def _square_zone(name="z", kind="generic"):
    return Zone(
        name=name,
        polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
        kind=kind,
    )


def test_point_inside_polygon():
    zone = _square_zone()
    assert zone.contains((5.0, 5.0)) is True


def test_point_outside_polygon():
    zone = _square_zone()
    assert zone.contains((50.0, 50.0)) is False


def test_point_outside_polygon_negative_coords():
    zone = _square_zone()
    assert zone.contains((-1.0, 5.0)) is False


def test_degenerate_polygon_never_contains():
    zone = Zone(name="bad", polygon=((0.0, 0.0), (1.0, 1.0)))
    assert zone.contains((0.5, 0.5)) is False


def test_zone_from_config_converts_types():
    config = ZoneConfig(name="entrance", polygon=[[0, 0], [10, 0], [10, 10], [0, 10]], kind="entrance")
    zone = Zone.from_config(config)
    assert zone.name == "entrance"
    assert zone.kind == "entrance"
    assert zone.polygon == ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))


def test_zone_manager_zones_containing_and_of_kind():
    entrance = _square_zone(name="entrance", kind="entrance")
    checkout = Zone(
        name="checkout",
        polygon=((20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0)),
        kind="checkout_counter",
    )
    manager = ZoneManager([entrance, checkout])

    assert [z.name for z in manager.zones_containing((5.0, 5.0))] == ["entrance"]
    assert [z.name for z in manager.zones_containing((25.0, 25.0))] == ["checkout"]
    assert manager.zones_containing((100.0, 100.0)) == []
    assert [z.name for z in manager.zones_of_kind("entrance")] == ["entrance"]
