from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from signal_noise.client import SignalClient

log = logging.getLogger(__name__)


class DataStore:
    """SQLite-backed data cache that syncs from SignalClient."""

    def __init__(self, db_path: Path, client: SignalClient | None = None):
        self._db_path = db_path
        self._client = client
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS signals ("
            "  name TEXT, date TEXT, value REAL,"
            "  PRIMARY KEY (name, date)"
            ")"
        )
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(signals)")}
        if "resolution" not in cols:
            self._conn.execute(
                "ALTER TABLE signals ADD COLUMN resolution TEXT DEFAULT '1d'"
            )
            self._conn.commit()

    def sync(self, signals: list[str], resolution: str = "1d") -> None:
        if self._client is None:
            log.info("No API client configured — skipping sync")
            return

        # Warn about stale upstream signals
        stale = self._client.stale_signals()
        stale_names = {s["name"] for s in stale}
        overlap = stale_names & set(signals)
        if overlap:
            log.warning("Stale upstream signals: %s", ", ".join(sorted(overlap)))

        # Determine per-signal max date for incremental sync
        max_dates: dict[str, str] = {}
        for name in signals:
            row = self._conn.execute(
                "SELECT MAX(date) FROM signals WHERE name = ?"
                " AND COALESCE(resolution, '1d') = ?",
                (name, resolution),
            ).fetchone()
            if row[0]:
                max_dates[name] = row[0]

        # Use earliest max_date as conservative global since
        global_since = min(max_dates.values()) if max_dates else None

        api_resolution = resolution if resolution != "1d" else None
        batch = self._client.get_batch(
            signals, since=global_since, columns=["timestamp", "value"],
            resolution=api_resolution,
        )

        for name, df in batch.items():
            if df.empty:
                continue

            rows = []
            for _, r in df.iterrows():
                ts = r["timestamp"]
                val = r["value"]
                if pd.isna(ts) or pd.isna(val):
                    continue
                if resolution == "1d":
                    date_str = str(ts.date()) if hasattr(ts, "date") else str(ts)[:10]
                else:
                    date_str = str(ts)
                rows.append((name, date_str, float(val), resolution))

            self._conn.executemany(
                "INSERT OR REPLACE INTO signals (name, date, value, resolution)"
                " VALUES (?, ?, ?, ?)",
                rows,
            )
        self._conn.commit()

    def get_matrix(
        self,
        signals: list[str],
        start: str | None = None,
        end: str | None = None,
        resolution: str = "1d",
    ) -> pd.DataFrame:
        placeholders = ",".join("?" for _ in signals)
        query = (
            f"SELECT name, date, value FROM signals"
            f" WHERE name IN ({placeholders})"
            f" AND COALESCE(resolution, '1d') = ?"
        )
        params: list[str] = list(signals) + [resolution]

        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)

        query += " ORDER BY date"

        df = pd.read_sql_query(query, self._conn, params=params)
        if df.empty:
            return pd.DataFrame(columns=signals)

        matrix = df.pivot(index="date", columns="name", values="value")
        matrix = matrix.reindex(columns=signals)
        matrix = matrix.ffill()
        matrix.index.name = "date"
        return matrix

    def get_prices(
        self,
        signal: str,
        start: str | None = None,
        end: str | None = None,
    ) -> np.ndarray:
        query = "SELECT value FROM signals WHERE name = ?"
        params: list[str] = [signal]

        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)

        query += " ORDER BY date"

        cursor = self._conn.execute(query, params)
        values = [row[0] for row in cursor.fetchall()]
        return np.array(values, dtype=np.float64)

    def close(self) -> None:
        self._conn.close()
