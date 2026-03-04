"""Alpha Registry — SQLite-backed metadata store for alpha lifecycle."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DATA_DIR


class AlphaState:
    BORN = "born"
    ACTIVE = "active"
    PROBATION = "probation"
    REJECTED = "rejected"
    DORMANT = "dormant"


@dataclass
class AlphaRecord:
    alpha_id: str
    expression: str
    state: str = AlphaState.BORN
    fitness: float = 0.0
    oos_sharpe: float = 0.0
    oos_log_growth: float = 0.0
    pbo: float = 1.0
    dsr_pvalue: float = 1.0
    turnover: float = 0.0
    correlation_avg: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: dict = field(default_factory=dict)

    _OOS_FITNESS_MAP = {"sharpe": "oos_sharpe", "log_growth": "oos_log_growth"}

    def oos_fitness(self, metric: str = "sharpe") -> float:
        return getattr(self, self._OOS_FITNESS_MAP[metric])


class AlphaRegistry:
    """SQLite registry for alpha factor metadata and lifecycle tracking."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or DATA_DIR / "alpha_registry.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS alphas (
                alpha_id TEXT PRIMARY KEY,
                expression TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'born',
                fitness REAL DEFAULT 0.0,
                oos_sharpe REAL DEFAULT 0.0,
                pbo REAL DEFAULT 1.0,
                dsr_pvalue REAL DEFAULT 1.0,
                turnover REAL DEFAULT 0.0,
                correlation_avg REAL DEFAULT 0.0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_state ON alphas(state)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_oos_sharpe ON alphas(oos_sharpe DESC)
        """)
        # Pipeline v2: candidate queue for evo daemon → validator flow
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY,
                expression TEXT NOT NULL,
                fitness REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                oos_sharpe REAL,
                pbo REAL,
                dsr_pvalue REAL,
                behavior_json TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                validated_at REAL,
                error_message TEXT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candidates_status
            ON candidates(status)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candidates_created
            ON candidates(created_at)
        """)
        # Pipeline v2: pre-computed diversity scores from validator
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS diversity_cache (
                alpha_id TEXT PRIMARY KEY,
                diversity_score REAL NOT NULL,
                computed_at REAL NOT NULL,
                n_alphas_compared INTEGER NOT NULL
            )
        """)
        # Migration: add oos_log_growth column if missing
        try:
            self._conn.execute(
                "ALTER TABLE alphas ADD COLUMN oos_log_growth REAL DEFAULT 0.0"
            )
        except sqlite3.OperationalError:
            pass
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_oos_log_growth
            ON alphas(oos_log_growth DESC)
        """)
        self._conn.commit()

    def register(self, record: AlphaRecord) -> None:
        now = time.time()
        if record.created_at == 0.0:
            record.created_at = now
        record.updated_at = now
        self._conn.execute(
            """INSERT OR REPLACE INTO alphas
               (alpha_id, expression, state, fitness, oos_sharpe, oos_log_growth,
                pbo, dsr_pvalue, turnover, correlation_avg,
                created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.alpha_id,
                record.expression,
                record.state,
                record.fitness,
                record.oos_sharpe,
                record.oos_log_growth,
                record.pbo,
                record.dsr_pvalue,
                record.turnover,
                record.correlation_avg,
                record.created_at,
                record.updated_at,
                json.dumps(record.metadata),
            ),
        )
        self._conn.commit()

    def get(self, alpha_id: str) -> AlphaRecord | None:
        row = self._conn.execute(
            "SELECT * FROM alphas WHERE alpha_id = ?", (alpha_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_by_state(self, state: str) -> list[AlphaRecord]:
        rows = self._conn.execute(
            "SELECT * FROM alphas WHERE state = ? ORDER BY oos_sharpe DESC",
            (state,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_active(self) -> list[AlphaRecord]:
        return self.list_by_state(AlphaState.ACTIVE)

    def update_state(self, alpha_id: str, new_state: str) -> None:
        self._conn.execute(
            "UPDATE alphas SET state = ?, updated_at = ? WHERE alpha_id = ?",
            (new_state, time.time(), alpha_id),
        )
        self._conn.commit()

    def update_metrics(
        self, alpha_id: str, oos_sharpe: float | None = None,
        pbo: float | None = None, dsr_pvalue: float | None = None,
    ) -> None:
        updates = []
        params: list = []
        if oos_sharpe is not None:
            updates.append("oos_sharpe = ?")
            params.append(oos_sharpe)
        if pbo is not None:
            updates.append("pbo = ?")
            params.append(pbo)
        if dsr_pvalue is not None:
            updates.append("dsr_pvalue = ?")
            params.append(dsr_pvalue)
        if not updates:
            return
        updates.append("updated_at = ?")
        params.append(time.time())
        params.append(alpha_id)
        self._conn.execute(
            f"UPDATE alphas SET {', '.join(updates)} WHERE alpha_id = ?",
            params,
        )
        self._conn.commit()

    def count(self, state: str | None = None) -> int:
        if state:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM alphas WHERE state = ?", (state,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM alphas").fetchone()
        return row[0]

    _ORDER_COLUMN = {"sharpe": "oos_sharpe", "log_growth": "oos_log_growth"}

    def top_trading(self, n: int = 30, metric: str = "sharpe") -> list[AlphaRecord]:
        """Top-N alphas by fitness metric from active + probation (for trading)."""
        col = self._ORDER_COLUMN[metric]
        rows = self._conn.execute(
            f"SELECT * FROM alphas WHERE state IN (?, ?) "
            f"ORDER BY {col} DESC LIMIT ?",
            (AlphaState.ACTIVE, AlphaState.PROBATION, n),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def top(self, n: int = 10, state: str | None = None, metric: str = "sharpe") -> list[AlphaRecord]:
        col = self._ORDER_COLUMN[metric]
        if state:
            rows = self._conn.execute(
                f"SELECT * FROM alphas WHERE state = ? ORDER BY {col} DESC LIMIT ?",
                (state, n),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT * FROM alphas ORDER BY {col} DESC LIMIT ?", (n,)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> AlphaRecord:
        keys = row.keys()
        return AlphaRecord(
            alpha_id=row["alpha_id"],
            expression=row["expression"],
            state=row["state"],
            fitness=row["fitness"],
            oos_sharpe=row["oos_sharpe"],
            oos_log_growth=row["oos_log_growth"] if "oos_log_growth" in keys else 0.0,
            pbo=row["pbo"],
            dsr_pvalue=row["dsr_pvalue"],
            turnover=row["turnover"],
            correlation_avg=row["correlation_avg"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]),
        )
