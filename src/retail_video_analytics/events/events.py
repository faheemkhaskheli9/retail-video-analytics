"""Zone/event logic: dwell time, entry/exit counting, cashier presence.

This module turns a stream of per-frame tracks (from the tracker) plus zone
definitions into the retail-relevant signals listed in the README's feature
list: customer counting, entry/exit counts, dwell time, heatmap samples,
cashier presence, and cashier-absence duration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..tracking.tracker import Track
from ..zones.zone import ZoneManager

ENTRANCE_KIND = "entrance"
CHECKOUT_KIND = "checkout_counter"


@dataclass
class DwellRecord:
    track_id: int
    zone_name: str
    seconds: float = 0.0
    entries: int = 0


@dataclass
class ZoneCounts:
    zone_name: str
    entries: int = 0
    exits: int = 0


@dataclass
class CashierAbsenceAlert:
    zone_name: str
    start_frame: int
    seconds: float


@dataclass
class FrameEvent:
    """Discrete events that happened on a single processed frame."""

    frame_index: int
    entries: list[tuple[str, int]] = field(default_factory=list)   # (zone_name, track_id)
    exits: list[tuple[str, int]] = field(default_factory=list)     # (zone_name, track_id)
    cashier_absent: list[str] = field(default_factory=list)        # zone_names newly flagged


class EventEngine:
    """Stateful accumulator that turns tracked positions into retail events."""

    def __init__(
        self,
        zone_manager: ZoneManager,
        seconds_per_frame: float,
        cashier_absence_seconds: float = 30.0,
    ) -> None:
        self.zones = zone_manager
        self.seconds_per_frame = seconds_per_frame
        self.cashier_absence_seconds = cashier_absence_seconds

        # zone_name -> set of track_ids currently inside
        self._occupancy: dict[str, set[int]] = {z.name: set() for z in self.zones.zones}
        # (track_id, zone_name) -> DwellRecord
        self._dwell: dict[tuple[int, str], DwellRecord] = {}
        # zone_name -> ZoneCounts
        self._counts: dict[str, ZoneCounts] = {
            z.name: ZoneCounts(zone_name=z.name) for z in self.zones.zones
        }
        # zone_name -> consecutive empty frames (for checkout zones)
        self._empty_streak: dict[str, int] = {z.name: 0 for z in self.zones.zones}
        self.cashier_alerts: list[CashierAbsenceAlert] = []
        self.max_concurrent_customers = 0
        self.heatmap_samples: list[tuple[float, float]] = []
        self._alerted_zones: set[str] = set()

        self._all_track_ids: set[int] = set()

    def process(self, frame_index: int, tracks: list[Track]) -> FrameEvent:
        event = FrameEvent(frame_index=frame_index)
        current_ids = {t.track_id for t in tracks}
        self._all_track_ids |= current_ids
        self.max_concurrent_customers = max(self.max_concurrent_customers, len(tracks))

        new_occupancy: dict[str, set[int]] = {name: set() for name in self._occupancy}
        for track in tracks:
            self.heatmap_samples.append(track.centroid)
            for zone in self.zones.zones_containing(track.centroid):
                new_occupancy[zone.name].add(track.track_id)
                key = (track.track_id, zone.name)
                record = self._dwell.get(key)
                if record is None:
                    record = DwellRecord(track_id=track.track_id, zone_name=zone.name)
                    self._dwell[key] = record
                if track.track_id not in self._occupancy[zone.name]:
                    record.entries += 1
                    if zone.kind == ENTRANCE_KIND:
                        self._counts[zone.name].entries += 1
                        event.entries.append((zone.name, track.track_id))
                record.seconds += self.seconds_per_frame

        # Exits: someone who was in a zone last frame but isn't now.
        for zone_name, previous_ids in self._occupancy.items():
            left = previous_ids - new_occupancy[zone_name]
            if left:
                zone = next(z for z in self.zones.zones if z.name == zone_name)
                for track_id in left:
                    if zone.kind == ENTRANCE_KIND:
                        self._counts[zone_name].exits += 1
                        event.exits.append((zone_name, track_id))

        self._occupancy = new_occupancy

        # Cashier / counter-presence absence tracking.
        for zone in self.zones.zones_of_kind(CHECKOUT_KIND):
            occupied = len(self._occupancy[zone.name]) > 0
            if occupied:
                self._empty_streak[zone.name] = 0
                self._alerted_zones.discard(zone.name)
            else:
                self._empty_streak[zone.name] += 1
                absence_seconds = self._empty_streak[zone.name] * self.seconds_per_frame
                if (
                    absence_seconds >= self.cashier_absence_seconds
                    and zone.name not in self._alerted_zones
                ):
                    self._alerted_zones.add(zone.name)
                    self.cashier_alerts.append(
                        CashierAbsenceAlert(
                            zone_name=zone.name,
                            start_frame=frame_index,
                            seconds=absence_seconds,
                        )
                    )
                    event.cashier_absent.append(zone.name)

        return event

    # -- summary accessors -------------------------------------------------

    def dwell_records(self) -> list[DwellRecord]:
        return list(self._dwell.values())

    def zone_counts(self) -> list[ZoneCounts]:
        return list(self._counts.values())

    def unique_customer_count(self) -> int:
        return len(self._all_track_ids)

    def average_dwell_seconds(self, zone_name: str) -> float:
        records = [r for r in self._dwell.values() if r.zone_name == zone_name]
        if not records:
            return 0.0
        return sum(r.seconds for r in records) / len(records)
