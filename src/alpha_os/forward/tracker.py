"""Forward tracker — SQLite-backed daily return persistence for forward-tested hypotheses."""
from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import (
    DATA_DIR,
    HYPOTHESIS_OBSERVATIONS_DB_NAME,
    LEGACY_FORWARD_RETURNS_DB_NAME,
)


@dataclass
class ForwardRecord:
    hypothesis_id: str
    date: str
    signal_value: float
    daily_return: float
    cumulative_return: float


@dataclass
class ForwardSummary:
    hypothesis_id: str
    forward_start_date: str
    n_days: int
    total_return: float
    sharpe: float
    max_dd: float
    last_date: str


class ForwardTracker:
    """Persist and query daily forward returns for adopted hypotheses."""

    _OBSERVATIONS_TABLE = "hypothesis_observations"
    _OBSERVATION_META_TABLE = "hypothesis_observation_meta"
    _LEGACY_OBSERVATIONS_TABLE = "forward_returns"
    _LEGACY_META_TABLE = "forward_meta"

    def __init__(self, db_path: Path | None = None):
        self._db_path = self._resolve_db_path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    @classmethod
    def _resolve_db_path(cls, db_path: Path | None) -> Path:
        resolved = db_path or DATA_DIR / HYPOTHESIS_OBSERVATIONS_DB_NAME
        if resolved.name == HYPOTHESIS_OBSERVATIONS_DB_NAME and not resolved.exists():
            legacy_path = resolved.with_name(LEGACY_FORWARD_RETURNS_DB_NAME)
            if legacy_path.exists():
                legacy_path.rename(resolved)
        return resolved

    def _create_tables(self) -> None:
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._OBSERVATIONS_TABLE} (
                hypothesis_id TEXT NOT NULL,
                date TEXT NOT NULL,
                signal_value REAL NOT NULL,
                daily_return REAL NOT NULL,
                cumulative_return REAL NOT NULL,
                recorded_at REAL NOT NULL,
                PRIMARY KEY (hypothesis_id, date)
            )
        """)
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._OBSERVATION_META_TABLE} (
                hypothesis_id TEXT PRIMARY KEY,
                forward_start_date TEXT NOT NULL,
                adopted_at REAL NOT NULL
            )
        """)
        self._migrate_schema()
        self._conn.commit()

    def _migrate_schema(self) -> None:
        self._migrate_table_name(
            legacy=self._LEGACY_OBSERVATIONS_TABLE,
            current=self._OBSERVATIONS_TABLE,
        )
        self._migrate_table_name(
            legacy=self._LEGACY_META_TABLE,
            current=self._OBSERVATION_META_TABLE,
        )
        for table in (self._OBSERVATIONS_TABLE, self._OBSERVATION_META_TABLE):
            columns = {
                row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")
            }
            if "alpha_id" in columns and "hypothesis_id" not in columns:
                self._conn.execute(
                    f"ALTER TABLE {table} RENAME COLUMN alpha_id TO hypothesis_id"
                )
        self._conn.execute("DROP INDEX IF EXISTS idx_fwd_alpha_date")
        self._conn.execute("DROP INDEX IF EXISTS idx_fwd_hypothesis_date")
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_hypothesis_observations_hypothesis_date
            ON {self._OBSERVATIONS_TABLE}(hypothesis_id, date)
            """
        )

    def _migrate_table_name(self, *, legacy: str, current: str) -> None:
        tables = {
            row["name"]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if legacy in tables and current in tables:
            columns = {
                row[1] for row in self._conn.execute(f"PRAGMA table_info({legacy})")
            }
            id_column = "hypothesis_id" if "hypothesis_id" in columns else "alpha_id"
            column_list = {
                self._LEGACY_OBSERVATIONS_TABLE: (
                    "date",
                    id_column,
                    "signal_value",
                    "daily_return",
                    "cumulative_return",
                    "recorded_at",
                ),
                self._LEGACY_META_TABLE: (
                    id_column,
                    "forward_start_date",
                    "adopted_at",
                ),
            }[legacy]
            target_columns = {
                self._LEGACY_OBSERVATIONS_TABLE: (
                    "date",
                    "hypothesis_id",
                    "signal_value",
                    "daily_return",
                    "cumulative_return",
                    "recorded_at",
                ),
                self._LEGACY_META_TABLE: (
                    "hypothesis_id",
                    "forward_start_date",
                    "adopted_at",
                ),
            }[legacy]
            self._conn.execute(
                f"""
                INSERT OR REPLACE INTO {current} ({", ".join(target_columns)})
                SELECT {", ".join(column_list)}
                FROM {legacy}
                """
            )
            self._conn.execute(f"DROP TABLE {legacy}")
            return
        if legacy in tables:
            self._conn.execute(f"ALTER TABLE {legacy} RENAME TO {current}")

    def register_hypothesis(self, hypothesis_id: str, start_date: str) -> None:
        self._conn.execute(
            f"INSERT OR IGNORE INTO {self._OBSERVATION_META_TABLE} "
            "(hypothesis_id, forward_start_date, adopted_at) "
            "VALUES (?, ?, ?)",
            (hypothesis_id, start_date, time.time()),
        )
        self._conn.commit()

    def get_hypothesis_start_date(self, hypothesis_id: str) -> str | None:
        row = self._conn.execute(
            f"SELECT forward_start_date FROM {self._OBSERVATION_META_TABLE} "
            "WHERE hypothesis_id = ?",
            (hypothesis_id,),
        ).fetchone()
        return row["forward_start_date"] if row else None

    def record(
        self,
        hypothesis_id: str,
        date: str,
        signal_value: float,
        daily_return: float,
    ) -> None:
        prev = self._conn.execute(
            f"SELECT cumulative_return FROM {self._OBSERVATIONS_TABLE} "
            "WHERE hypothesis_id = ? AND date < ? ORDER BY date DESC LIMIT 1",
            (hypothesis_id, date),
        ).fetchone()
        prev_cum = prev["cumulative_return"] if prev else 1.0
        cum = prev_cum * (1.0 + daily_return)

        self._conn.execute(
            f"INSERT OR REPLACE INTO {self._OBSERVATIONS_TABLE} "
            "(hypothesis_id, date, signal_value, daily_return, cumulative_return, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (hypothesis_id, date, signal_value, daily_return, cum, time.time()),
        )
        self._conn.commit()

    def get_hypothesis_returns(self, hypothesis_id: str) -> list[float]:
        rows = self._conn.execute(
            f"SELECT daily_return FROM {self._OBSERVATIONS_TABLE} "
            "WHERE hypothesis_id = ? ORDER BY date",
            (hypothesis_id,),
        ).fetchall()
        return [row["daily_return"] for row in rows]

    def get_hypothesis_realizable_returns(
        self,
        hypothesis_id: str,
        *,
        supports_short: bool,
    ) -> list[float]:
        if supports_short:
            return self.get_hypothesis_returns(hypothesis_id)
        rows = self._conn.execute(
            f"SELECT signal_value, daily_return FROM {self._OBSERVATIONS_TABLE} "
            "WHERE hypothesis_id = ? ORDER BY date",
            (hypothesis_id,),
        ).fetchall()
        realized: list[float] = []
        for row in rows:
            signal_value = float(row["signal_value"])
            daily_return = float(row["daily_return"])
            if not math.isfinite(signal_value) or not math.isfinite(daily_return):
                realized.append(0.0)
                continue
            realized.append(daily_return if signal_value > 0.0 else 0.0)
        return realized

    def get_hypothesis_records(self, hypothesis_id: str) -> list[ForwardRecord]:
        rows = self._conn.execute(
            f"SELECT * FROM {self._OBSERVATIONS_TABLE} WHERE hypothesis_id = ? ORDER BY date",
            (hypothesis_id,),
        ).fetchall()
        return [
            ForwardRecord(
                hypothesis_id=row["hypothesis_id"],
                date=row["date"],
                signal_value=row["signal_value"],
                daily_return=row["daily_return"],
                cumulative_return=row["cumulative_return"],
            )
            for row in rows
        ]

    def get_hypothesis_last_date(self, hypothesis_id: str) -> str | None:
        row = self._conn.execute(
            f"SELECT MAX(date) as last_date FROM {self._OBSERVATIONS_TABLE} "
            "WHERE hypothesis_id = ?",
            (hypothesis_id,),
        ).fetchone()
        return row["last_date"] if row and row["last_date"] else None

    def hypothesis_summary(self, hypothesis_id: str) -> ForwardSummary | None:
        start_date = self.get_hypothesis_start_date(hypothesis_id)
        if start_date is None:
            return None
        returns = self.get_hypothesis_returns(hypothesis_id)
        if not returns:
            return ForwardSummary(
                hypothesis_id=hypothesis_id,
                forward_start_date=start_date,
                n_days=0,
                total_return=0.0,
                sharpe=0.0,
                max_dd=0.0,
                last_date=start_date,
            )
        rets = np.array(returns)
        std = rets.std()
        sharpe = float(rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0
        cum = np.cumprod(1.0 + rets)
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum) / peak
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0
        total_return = float(cum[-1] - 1.0)
        last_date = self.get_hypothesis_last_date(hypothesis_id) or start_date

        return ForwardSummary(
            hypothesis_id=hypothesis_id,
            forward_start_date=start_date,
            n_days=len(returns),
            total_return=total_return,
            sharpe=sharpe,
            max_dd=max_dd,
            last_date=last_date,
        )

    def tracked_hypothesis_ids(self) -> list[str]:
        rows = self._conn.execute(
            f"SELECT hypothesis_id FROM {self._OBSERVATION_META_TABLE} "
            "ORDER BY hypothesis_id"
        ).fetchall()
        return [row["hypothesis_id"] for row in rows]

    def close(self) -> None:
        self._conn.close()
