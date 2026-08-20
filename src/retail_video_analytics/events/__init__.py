"""Zone/event logic: dwell time, entry/exit counting, cashier presence."""

from .events import (
    CHECKOUT_KIND,
    ENTRANCE_KIND,
    CashierAbsenceAlert,
    DwellRecord,
    EventEngine,
    FrameEvent,
    ZoneCounts,
)

__all__ = [
    "CHECKOUT_KIND",
    "ENTRANCE_KIND",
    "CashierAbsenceAlert",
    "DwellRecord",
    "EventEngine",
    "FrameEvent",
    "ZoneCounts",
]
