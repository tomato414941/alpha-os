"""Forward tracker — SQLite-backed daily return persistence for forward-tested hypotheses."""
from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import DATA_DIR


@dataclass
class ForwardRecord:
    hypothesis_id: str
    date: str
    signal_value: float
    daily_return: float
    cumulative_return: float

    @property
    def alpha_id(self) -> str:
        return self.hypothesis_id


@dataclass
class ForwardSummary:
    hypothesis_id: str
    forward_start_date: str
    n_days: int
    total_return: float
    sharpe: float
    max_dd: float
    last_date: str

    @property
    def alpha_id(self) -> str:
        return self.hypothesis_id


class ForwardTracker:
    """Persist and query daily forward returns for adopted hypotheses."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or DATA_DIR / "forward_returns.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS forward_returns (
                alpha_id TEXT NOT NULL,
                date TEXT NOT NULL,
                signal_value REAL NOT NULL,
                daily_return REAL NOT NULL,
                cumulative_return REAL NOT NULL,
                recorded_at REAL NOT NULL,
                PRIMARY KEY (alpha_id, date)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS forward_meta (
                alpha_id TEXT PRIMARY KEY,
                forward_start_date TEXT NOT NULL,
                adopted_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fwd_alpha_date
            ON forward_returns(alpha_id, date)
        """)
        self._conn.commit()

    def register_hypothesis(self, hypothesis_id: str, start_date: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO forward_meta (alpha_id, forward_start_date, adopted_at) "
            "VALUES (?, ?, ?)",
            (hypothesis_id, start_date, time.time()),
        )
        self._conn.commit()

    def register_alpha(self, alpha_id: str, start_date: str) -> None:
        self.register_hypothesis(alpha_id, start_date)

    def get_hypothesis_start_date(self, hypothesis_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT forward_start_date FROM forward_meta WHERE alpha_id = ?",
            (hypothesis_id,),
        ).fetchone()
        return row["forward_start_date"] if row else None

    def get_start_date(self, alpha_id: str) -> str | None:
        return self.get_hypothesis_start_date(alpha_id)

    def record(
        self,
        hypothesis_id: str,
        date: str,
        signal_value: float,
        daily_return: float,
    ) -> None:
        prev = self._conn.execute(
            "SELECT cumulative_return FROM forward_returns "
            "WHERE alpha_id = ? AND date < ? ORDER BY date DESC LIMIT 1",
            (hypothesis_id, date),
        ).fetchone()
        prev_cum = prev["cumulative_return"] if prev else 1.0
        cum = prev_cum * (1.0 + daily_return)

        self._conn.execute(
            "INSERT OR REPLACE INTO forward_returns "
            "(alpha_id, date, signal_value, daily_return, cumulative_return, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (hypothesis_id, date, signal_value, daily_return, cum, time.time()),
        )
        self._conn.commit()

    def get_hypothesis_returns(self, hypothesis_id: str) -> list[float]:
        rows = self._conn.execute(
            "SELECT daily_return FROM forward_returns "
            "WHERE alpha_id = ? ORDER BY date",
            (hypothesis_id,),
        ).fetchall()
        return [row["daily_return"] for row in rows]

    def get_returns(self, alpha_id: str) -> list[float]:
        return self.get_hypothesis_returns(alpha_id)

    def get_hypothesis_realizable_returns(
        self,
        hypothesis_id: str,
        *,
        supports_short: bool,
    ) -> list[float]:
        if supports_short:
            return self.get_hypothesis_returns(hypothesis_id)
        rows = self._conn.execute(
            "SELECT signal_value, daily_return FROM forward_returns "
            "WHERE alpha_id = ? ORDER BY date",
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

    def get_realizable_returns(
        self,
        alpha_id: str,
        *,
        supports_short: bool,
    ) -> list[float]:
        return self.get_hypothesis_realizable_returns(
            alpha_id,
            supports_short=supports_short,
        )

    def get_hypothesis_records(self, hypothesis_id: str) -> list[ForwardRecord]:
        rows = self._conn.execute(
            "SELECT * FROM forward_returns WHERE alpha_id = ? ORDER BY date",
            (hypothesis_id,),
        ).fetchall()
        return [
            ForwardRecord(
                hypothesis_id=row["alpha_id"],
                date=row["date"],
                signal_value=row["signal_value"],
                daily_return=row["daily_return"],
                cumulative_return=row["cumulative_return"],
            )
            for row in rows
        ]

    def get_records(self, alpha_id: str) -> list[ForwardRecord]:
        return self.get_hypothesis_records(alpha_id)

    def get_hypothesis_last_date(self, hypothesis_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT MAX(date) as last_date FROM forward_returns WHERE alpha_id = ?",
            (hypothesis_id,),
        ).fetchone()
        return row["last_date"] if row and row["last_date"] else None

    def get_last_date(self, alpha_id: str) -> str | None:
        return self.get_hypothesis_last_date(alpha_id)

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

    def summary(self, alpha_id: str) -> ForwardSummary | None:
        return self.hypothesis_summary(alpha_id)

    def tracked_hypothesis_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT alpha_id FROM forward_meta ORDER BY alpha_id"
        ).fetchall()
        return [row["alpha_id"] for row in rows]

    def tracked_alpha_ids(self) -> list[str]:
        return self.tracked_hypothesis_ids()

    def close(self) -> None:
        self._conn.close()
