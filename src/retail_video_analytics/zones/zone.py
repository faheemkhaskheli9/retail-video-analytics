"""Zone definitions and point-in-polygon membership testing.

Zones are simple polygons in frame pixel coordinates, configured in YAML
(see ``configs/default.yaml``). ``kind`` tags a zone's role (e.g.
``"entrance"`` for entry/exit counting, ``"checkout_counter"`` for
cashier-presence tracking) so the event logic knows how to interpret
occupancy of that zone.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ZoneConfig


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: tuple[tuple[float, float], ...]
    kind: str = "generic"

    @classmethod
    def from_config(cls, config: ZoneConfig) -> "Zone":
        return cls(
            name=config.name,
            polygon=tuple((float(p[0]), float(p[1])) for p in config.polygon),
            kind=config.kind,
        )

    def contains(self, point: tuple[float, float]) -> bool:
        """Ray-casting point-in-polygon test."""
        x, y = point
        n = len(self.polygon)
        if n < 3:
            return False
        inside = False
        x1, y1 = self.polygon[-1]
        for x2, y2 in self.polygon:
            if (y1 > y) != (y2 > y):
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_intersect:
                    inside = not inside
            x1, y1 = x2, y2
        return inside


class ZoneManager:
    """Holds all configured zones and reports which zones a point falls in."""

    def __init__(self, zones: list[Zone]) -> None:
        self.zones = zones

    @classmethod
    def from_config(cls, zone_configs: list[ZoneConfig]) -> "ZoneManager":
        return cls([Zone.from_config(z) for z in zone_configs])

    def zones_containing(self, point: tuple[float, float]) -> list[Zone]:
        return [z for z in self.zones if z.contains(point)]

    def zones_of_kind(self, kind: str) -> list[Zone]:
        return [z for z in self.zones if z.kind == kind]
