"""Analytics storage.

The README's tech stack lists PostgreSQL for a production deployment; this
MVP uses SQLite (stdlib ``sqlite3``, zero extra services to stand up) behind
the same simple repository API so swapping the backend later is a matter of
changing the connection string, not the calling code.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from ..events.events import CashierAbsenceAlert, DwellRecord, ZoneCounts

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_source TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    frame_count INTEGER NOT NULL,
    unique_customers INTEGER NOT NULL,
    max_concurrent_customers INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS zone_counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    zone_name TEXT NOT NULL,
    entries INTEGER NOT NULL,
    exits INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS dwell_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    track_id INTEGER NOT NULL,
    zone_name TEXT NOT NULL,
    seconds REAL NOT NULL,
    entries INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS cashier_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    zone_name TEXT NOT NULL,
    start_frame INTEGER NOT NULL,
    seconds REAL NOT NULL
);
"""


class AnalyticsStore:
    """Thin repository around a SQLite database for run results."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AnalyticsStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def save_run(
        self,
        video_source: str,
        frame_count: int,
        unique_customers: int,
        max_concurrent_customers: int,
        zone_counts: list[ZoneCounts],
        dwell_records: list[DwellRecord],
        cashier_alerts: list[CashierAbsenceAlert],
    ) -> int:
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "INSERT INTO runs (video_source, frame_count, unique_customers, "
                "max_concurrent_customers) VALUES (?, ?, ?, ?)",
                (video_source, frame_count, unique_customers, max_concurrent_customers),
            )
            run_id = cur.lastrowid

            for zc in zone_counts:
                cur.execute(
                    "INSERT INTO zone_counts (run_id, zone_name, entries, exits) "
                    "VALUES (?, ?, ?, ?)",
                    (run_id, zc.zone_name, zc.entries, zc.exits),
                )
            for dr in dwell_records:
                cur.execute(
                    "INSERT INTO dwell_records (run_id, track_id, zone_name, seconds, entries) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run_id, dr.track_id, dr.zone_name, dr.seconds, dr.entries),
                )
            for alert in cashier_alerts:
                cur.execute(
                    "INSERT INTO cashier_alerts (run_id, zone_name, start_frame, seconds) "
                    "VALUES (?, ?, ?, ?)",
                    (run_id, alert.zone_name, alert.start_frame, alert.seconds),
                )
            self._conn.commit()
            return run_id

    def get_run(self, run_id: int) -> dict:
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "SELECT id, video_source, started_at, frame_count, unique_customers, "
                "max_concurrent_customers FROM runs WHERE id = ?",
                (run_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"No run with id {run_id}")
            columns = [
                "id",
                "video_source",
                "started_at",
                "frame_count",
                "unique_customers",
                "max_concurrent_customers",
            ]
            return dict(zip(columns, row))

    def list_zone_counts(self, run_id: int) -> list[dict]:
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "SELECT zone_name, entries, exits FROM zone_counts WHERE run_id = ?",
                (run_id,),
            )
            return [
                {"zone_name": r[0], "entries": r[1], "exits": r[2]} for r in cur.fetchall()
            ]
