"""Prediction store — the coupling point between signal producers and the pipeline.

Producers write predictions. The pipeline reads and scores them.
Neither knows about the other.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


from ..config import DATA_DIR

_DEFAULT_DB = DATA_DIR / "predictions.db"


@dataclass(frozen=True)
class Prediction:
    signal_id: str
    date: str
    asset: str
    value: float
    horizon: int = 1
    recorded_at: float = 0.0


@dataclass(frozen=True)
class SignalMeta:
    signal_id: str
    source: str
    definition: str
    horizon: int = 1
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


class PredictionStore:
    """SQLite store for signal predictions.

    Producers call write() to submit predictions.
    The pipeline calls read() to get predictions for scoring.
    """

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or _DEFAULT_DB
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                signal_id   TEXT NOT NULL,
                date        TEXT NOT NULL,
                asset       TEXT NOT NULL,
                value       REAL NOT NULL,
                horizon     INTEGER NOT NULL DEFAULT 1,
                recorded_at REAL NOT NULL,
                PRIMARY KEY (signal_id, date, asset)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_date
            ON predictions(date DESC)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_signal
            ON predictions(signal_id, date DESC)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id   TEXT PRIMARY KEY,
                source      TEXT NOT NULL,
                definition  TEXT NOT NULL,
                horizon     INTEGER NOT NULL DEFAULT 1,
                metadata    TEXT DEFAULT '{}',
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            )
        """)
        self._conn.commit()

    # ── Producer API ──

    def register_signal(self, meta: SignalMeta) -> None:
        """Register a signal source. Idempotent — updates on conflict."""
        now = time.time()
        self._conn.execute(
            """INSERT INTO signals (signal_id, source, definition, horizon, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(signal_id) DO UPDATE SET
                 source=excluded.source, definition=excluded.definition,
                 horizon=excluded.horizon, metadata=excluded.metadata,
                 updated_at=excluded.updated_at""",
            (meta.signal_id, meta.source, meta.definition, meta.horizon,
             json.dumps(meta.metadata or {}), now, now),
        )
        self._conn.commit()

    def write(self, predictions: list[Prediction]) -> int:
        """Write predictions. Returns count of new rows inserted."""
        if not predictions:
            return 0
        self._conn.executemany(
            """INSERT OR REPLACE INTO predictions
               (signal_id, date, asset, value, horizon, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(p.signal_id, p.date, p.asset, p.value, p.horizon, p.recorded_at or time.time())
             for p in predictions],
        )
        self._conn.commit()
        return len(predictions)

    # ── Pipeline API ──

    def read_latest(self, date: str, assets: list[str] | None = None) -> dict[str, dict[str, float]]:
        """Read latest predictions for a given date.

        Returns {signal_id: {asset: value}}.
        """
        if assets:
            placeholders = ",".join("?" for _ in assets)
            rows = self._conn.execute(
                f"SELECT signal_id, asset, value FROM predictions "
                f"WHERE date = ? AND asset IN ({placeholders})",
                (date, *assets),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT signal_id, asset, value FROM predictions WHERE date = ?",
                (date,),
            ).fetchall()

        result: dict[str, dict[str, float]] = {}
        for signal_id, asset, value in rows:
            result.setdefault(signal_id, {})[asset] = value
        return result

    def read_signal_history(
        self, signal_id: str, asset: str, n_days: int = 60,
    ) -> list[tuple[str, float]]:
        """Read recent prediction history for one signal+asset pair.

        Returns [(date, value), ...] ordered by date descending.
        """
        rows = self._conn.execute(
            "SELECT date, value FROM predictions "
            "WHERE signal_id = ? AND asset = ? "
            "ORDER BY date DESC LIMIT ?",
            (signal_id, asset, n_days),
        ).fetchall()
        return rows

    def list_signals(self, source: str | None = None) -> list[SignalMeta]:
        """List registered signals."""
        if source:
            rows = self._conn.execute(
                "SELECT signal_id, source, definition, horizon, metadata "
                "FROM signals WHERE source = ? ORDER BY created_at DESC",
                (source,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT signal_id, source, definition, horizon, metadata "
                "FROM signals ORDER BY created_at DESC",
            ).fetchall()
        return [
            SignalMeta(
                signal_id=r[0], source=r[1], definition=r[2],
                horizon=r[3], metadata=json.loads(r[4]),
            )
            for r in rows
        ]

    def signal_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]

    def prediction_count(self, date: str | None = None) -> int:
        if date:
            return self._conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE date = ?", (date,),
            ).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
