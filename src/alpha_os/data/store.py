from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpha_os.data.client import SignalClient

log = logging.getLogger(__name__)


class DataStore:
    """SQLite-backed data cache that syncs from SignalClient."""

    def __init__(self, db_path: Path, client: SignalClient | None = None):
        self._db_path = db_path
        self._client = client
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS signals ("
            "  name TEXT, date TEXT, value REAL,"
            "  PRIMARY KEY (name, date)"
            ")"
        )
        self._conn.commit()

    def sync(self, signals: list[str]) -> None:
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
                "SELECT MAX(date) FROM signals WHERE name = ?", (name,)
            ).fetchone()
            if row[0]:
                max_dates[name] = row[0]

        # Use earliest max_date as conservative global since
        global_since = min(max_dates.values()) if max_dates else None

        batch = self._client.get_batch(
            signals, since=global_since, columns=["timestamp", "value"],
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
                date_str = str(ts.date()) if hasattr(ts, "date") else str(ts)[:10]
                rows.append((name, date_str, float(val)))

            self._conn.executemany(
                "INSERT OR REPLACE INTO signals (name, date, value) VALUES (?, ?, ?)",
                rows,
            )
        self._conn.commit()

    def import_from_signal_noise(self, source_db: Path, signals: list[str]) -> int:
        """Bulk-import daily data from signal-noise SQLite DB."""
        if not source_db.exists():
            log.warning("signal-noise DB not found: %s", source_db)
            return 0
        src = sqlite3.connect(str(source_db))
        src.execute("ATTACH DATABASE ? AS dst", (str(self._db_path),))

        placeholders = ",".join("?" for _ in signals)
        # signal-noise stores timestamp as ISO string; extract date part
        src.execute(
            f"INSERT OR REPLACE INTO dst.signals (name, date, value) "
            f"SELECT name, SUBSTR(timestamp, 1, 10), value "
            f"FROM signals WHERE name IN ({placeholders})",
            signals,
        )
        count = src.execute("SELECT changes()").fetchone()[0]
        src.commit()
        src.close()
        self._conn = sqlite3.connect(str(self._db_path))
        log.info("Imported %d rows from signal-noise", count)
        return count

    def get_matrix(
        self,
        signals: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        placeholders = ",".join("?" for _ in signals)
        query = f"SELECT name, date, value FROM signals WHERE name IN ({placeholders})"
        params: list[str] = list(signals)

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
