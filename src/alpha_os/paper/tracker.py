"""Paper trading portfolio tracker — SQLite persistence."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import DATA_DIR
from ..execution.executor import Fill


@dataclass
class PortfolioSnapshot:
    date: str
    cash: float
    positions: dict[str, float]
    portfolio_value: float
    daily_pnl: float
    daily_return: float
    combined_signal: float
    dd_scale: float
    vol_scale: float


@dataclass
class PaperTradingSummary:
    start_date: str
    end_date: str
    n_days: int
    initial_value: float
    final_value: float
    total_return: float
    sharpe: float
    max_drawdown: float
    total_trades: int
    current_positions: dict[str, float]
    current_cash: float


class PaperPortfolioTracker:
    """SQLite-backed persistence for paper trading state."""

    def __init__(self, db_path: Path | None = None):
        self._path = db_path or DATA_DIR / "paper_trading.db"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                date TEXT PRIMARY KEY,
                cash REAL NOT NULL,
                positions_json TEXT NOT NULL,
                portfolio_value REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                daily_return REAL NOT NULL,
                combined_signal REAL NOT NULL,
                dd_scale REAL NOT NULL,
                vol_scale REAL NOT NULL,
                recorded_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                order_id TEXT NOT NULL,
                recorded_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_fills_date ON paper_fills(date);
            CREATE TABLE IF NOT EXISTS alpha_signals (
                date TEXT NOT NULL,
                alpha_id TEXT NOT NULL,
                signal_value REAL NOT NULL,
                PRIMARY KEY (date, alpha_id)
            );
        """)

    def save_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO portfolio_snapshots
            (date, cash, positions_json, portfolio_value, daily_pnl,
             daily_return, combined_signal, dd_scale, vol_scale, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot.date,
                snapshot.cash,
                json.dumps(snapshot.positions),
                snapshot.portfolio_value,
                snapshot.daily_pnl,
                snapshot.daily_return,
                snapshot.combined_signal,
                snapshot.dd_scale,
                snapshot.vol_scale,
                time.time(),
            ),
        )
        self._conn.commit()

    def save_fills(self, date: str, fills: list[Fill]) -> None:
        now = time.time()
        rows = [
            (date, f.symbol, f.side, f.qty, f.price, f.order_id, now)
            for f in fills
        ]
        self._conn.executemany(
            """INSERT INTO paper_fills
            (date, symbol, side, qty, price, order_id, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def save_alpha_signals(
        self, date: str, signals: dict[str, float]
    ) -> None:
        rows = [(date, aid, val) for aid, val in signals.items()]
        self._conn.executemany(
            """INSERT OR REPLACE INTO alpha_signals
            (date, alpha_id, signal_value) VALUES (?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def get_last_snapshot(self) -> PortfolioSnapshot | None:
        row = self._conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return self._row_to_snapshot(row)

    def get_snapshot(self, date: str) -> PortfolioSnapshot | None:
        row = self._conn.execute(
            "SELECT * FROM portfolio_snapshots WHERE date = ?", (date,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_snapshot(row)

    def get_all_snapshots(self) -> list[PortfolioSnapshot]:
        rows = self._conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY date"
        ).fetchall()
        return [self._row_to_snapshot(r) for r in rows]

    def get_returns(self) -> list[float]:
        rows = self._conn.execute(
            "SELECT daily_return FROM portfolio_snapshots ORDER BY date"
        ).fetchall()
        return [r["daily_return"] for r in rows]

    def get_equity_curve(self) -> list[tuple[str, float]]:
        rows = self._conn.execute(
            "SELECT date, portfolio_value FROM portfolio_snapshots ORDER BY date"
        ).fetchall()
        return [(r["date"], r["portfolio_value"]) for r in rows]

    def get_total_trades(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM paper_fills").fetchone()
        return row["cnt"] if row else 0

    def summary(self) -> PaperTradingSummary | None:
        snapshots = self._conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY date"
        ).fetchall()
        if not snapshots:
            return None

        first = snapshots[0]
        last = snapshots[-1]
        initial = first["portfolio_value"] - first["daily_pnl"]
        if initial <= 0:
            initial = first["portfolio_value"]
        final = last["portfolio_value"]
        total_ret = (final - initial) / initial if initial > 0 else 0.0

        returns = np.array([s["daily_return"] for s in snapshots])
        std = float(np.std(returns))
        sharpe = float(np.mean(returns) / std * np.sqrt(252)) if std > 1e-10 else 0.0

        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        dd = (peak - cumulative) / np.where(peak > 0, peak, 1.0)
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        return PaperTradingSummary(
            start_date=first["date"],
            end_date=last["date"],
            n_days=len(snapshots),
            initial_value=initial,
            final_value=final,
            total_return=total_ret,
            sharpe=sharpe,
            max_drawdown=max_dd,
            total_trades=self.get_total_trades(),
            current_positions=json.loads(last["positions_json"]),
            current_cash=last["cash"],
        )

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            date=row["date"],
            cash=row["cash"],
            positions=json.loads(row["positions_json"]),
            portfolio_value=row["portfolio_value"],
            daily_pnl=row["daily_pnl"],
            daily_return=row["daily_return"],
            combined_signal=row["combined_signal"],
            dd_scale=row["dd_scale"],
            vol_scale=row["vol_scale"],
        )
