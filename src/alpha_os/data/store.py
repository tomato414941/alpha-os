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
    """SQLite-backed data cache that syncs from signal-noise API.

    Sync strategy:
    - New signals (not in cache): full fetch (since=None)
    - Existing signals with short history: backfill from earliest available
    - Existing signals with sufficient history: incremental (since=MAX(date))
    """

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
            "  resolution TEXT DEFAULT '1d',"
            "  PRIMARY KEY (name, date)"
            ")"
        )
        self._conn.commit()

    def sync(
        self,
        signals: list[str],
        resolution: str = "1d",
        *,
        min_history_days: int = 0,
    ) -> None:
        """Sync signals from signal-noise API.

        For each signal:
        - Not in cache → full fetch (since=None)
        - In cache but fewer rows than min_history_days → full re-fetch
        - Otherwise → incremental (since=MAX(date))
        """
        if self._client is None:
            log.info("No API client configured — skipping sync")
            return

        # Query local state for all signals at once
        local_state = self._get_local_state(signals, resolution)

        # Partition into full-fetch vs incremental
        full_fetch: list[str] = []
        incremental: dict[str | None, list[str]] = {}

        for name in signals:
            state = local_state.get(name)
            if state is None:
                # New signal — full fetch
                full_fetch.append(name)
            elif min_history_days > 0 and state["count"] < min_history_days:
                # Insufficient history — full re-fetch
                full_fetch.append(name)
            else:
                # Incremental from last date
                since = state["max_date"]
                incremental.setdefault(since, []).append(name)

        api_resolution = resolution if resolution != "1d" else None

        # Full fetches (since=None)
        if full_fetch:
            log.info("Full sync: %d signals (new or short history)", len(full_fetch))
            self._fetch_and_store(full_fetch, since=None, resolution=resolution,
                                  api_resolution=api_resolution)

        # Incremental fetches (grouped by since date)
        for since, batch_names in incremental.items():
            self._fetch_and_store(batch_names, since=since, resolution=resolution,
                                  api_resolution=api_resolution)

        self._conn.commit()

    def _get_local_state(
        self, signals: list[str], resolution: str,
    ) -> dict[str, dict]:
        """Query MIN(date), MAX(date), COUNT(*) for each signal."""
        result: dict[str, dict] = {}
        for chunk_start in range(0, len(signals), 500):
            chunk = signals[chunk_start:chunk_start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT name, MIN(date), MAX(date), COUNT(*) FROM signals"
                f" WHERE name IN ({placeholders})"
                f" AND COALESCE(resolution, '1d') = ?"
                f" GROUP BY name",
                chunk + [resolution],
            ).fetchall()
            for name, min_date, max_date, count in rows:
                result[name] = {
                    "min_date": min_date,
                    "max_date": max_date,
                    "count": count,
                }
        return result

    def _fetch_and_store(
        self,
        names: list[str],
        since: str | None,
        resolution: str,
        api_resolution: str | None,
    ) -> None:
        """Fetch a batch of signals from API and store in SQLite."""
        batch = self._client.get_batch(
            names,
            since=since,
            columns=["timestamp", "value"],
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
            if rows:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO signals (name, date, value, resolution)"
                    " VALUES (?, ?, ?, ?)",
                    rows,
                )

    def get_matrix(
        self,
        signals: list[str],
        start: str | None = None,
        end: str | None = None,
        resolution: str = "1d",
        ffill_limit: int = 5,
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
        matrix = matrix.ffill(limit=ffill_limit)
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

    def signal_row_counts(self, signals: list[str]) -> dict[str, int]:
        """Return {name: row_count} for the given signals."""
        result: dict[str, int] = {}
        for chunk_start in range(0, len(signals), 500):
            chunk = signals[chunk_start:chunk_start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT name, COUNT(*) FROM signals"
                f" WHERE name IN ({placeholders}) GROUP BY name",
                chunk,
            ).fetchall()
            for name, count in rows:
                result[name] = count
        return result

    def close(self) -> None:
        self._conn.close()
