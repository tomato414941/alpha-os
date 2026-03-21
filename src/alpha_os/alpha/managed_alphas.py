"""Managed alpha store backed by SQLite."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DATA_DIR
from .admission_queue import CandidateSeed, queue_candidate_expressions, queue_candidates


class AlphaState:
    CANDIDATE = "candidate"
    ACTIVE = "active"
    DORMANT = "dormant"
    REJECTED = "rejected"

    _CANONICAL = {
        CANDIDATE: CANDIDATE,
        ACTIVE: ACTIVE,
        DORMANT: DORMANT,
        REJECTED: REJECTED,
    }

    @classmethod
    def canonical(cls, state: str) -> str:
        return cls._CANONICAL.get(state, state)



@dataclass
class AlphaRecord:
    alpha_id: str
    expression: str
    state: str = AlphaState.CANDIDATE
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
    stake: float = 0.0

    _OOS_FITNESS_MAP = {"sharpe": "oos_sharpe", "log_growth": "oos_log_growth"}

    def oos_fitness(self, metric: str = "sharpe") -> float:
        return getattr(self, self._OOS_FITNESS_MAP[metric])


@dataclass(frozen=True)
class DeployedAlphaEntry:
    alpha_id: str
    slot: int
    deployed_at: float
    deployment_score: float = 0.0
    metadata: dict = field(default_factory=dict)


class ManagedAlphaStore:
    """SQLite store for managed alpha metadata and lifecycle tracking."""

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
                state TEXT NOT NULL DEFAULT 'candidate',
                fitness REAL DEFAULT 0.0,
                oos_sharpe REAL DEFAULT 0.0,
                pbo REAL DEFAULT 1.0,
                dsr_pvalue REAL DEFAULT 1.0,
                turnover REAL DEFAULT 0.0,
                correlation_avg REAL DEFAULT 0.0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}',
                stake REAL DEFAULT 0.0
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_state ON alphas(state)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_oos_sharpe ON alphas(oos_sharpe DESC)
        """)
        # Migrate: add stake column to existing tables
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(alphas)")}
        if "stake" not in cols:
            self._conn.execute("ALTER TABLE alphas ADD COLUMN stake REAL DEFAULT 0.0")
            self._conn.execute(
                "UPDATE alphas SET stake = MAX(oos_sharpe, 0.0) WHERE state = 'active'"
            )
            self._conn.commit()
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_stake ON alphas(stake DESC)
        """)
        # Pipeline v2: candidate queue for alpha generator → admission flow
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY,
                source TEXT NOT NULL DEFAULT '',
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
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS deployed_alphas (
                alpha_id TEXT PRIMARY KEY,
                slot INTEGER NOT NULL UNIQUE,
                deployed_at REAL NOT NULL,
                deployment_score REAL NOT NULL DEFAULT 0.0,
                metadata TEXT DEFAULT '{}'
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_deployed_alphas_slot
            ON deployed_alphas(slot)
        """)
        # Migration: add oos_log_growth column if missing
        try:
            self._conn.execute(
                "ALTER TABLE alphas ADD COLUMN oos_log_growth REAL DEFAULT 0.0"
            )
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute(
                "ALTER TABLE candidates ADD COLUMN source TEXT NOT NULL DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass
        self._conn.execute(
            "UPDATE candidates SET source = 'alpha_generator_btc' "
            "WHERE source = '' AND candidate_id LIKE 'alpha_generator_btc_%'"
        )
        self._conn.execute(
            "UPDATE candidates SET source = 'manual' "
            "WHERE source = '' AND candidate_id LIKE 'manual_%'"
        )
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candidates_source_status_created
            ON candidates(source, status, created_at DESC)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_oos_log_growth
            ON alphas(oos_log_growth DESC)
        """)
        self._conn.commit()

    def replace_all(self, records: list[AlphaRecord]) -> None:
        values = [self._record_values(record) for record in records]
        self._conn.execute("DELETE FROM deployed_alphas")
        self._conn.execute("DELETE FROM alphas")
        if values:
            self._conn.executemany(
                """INSERT INTO alphas
                   (alpha_id, expression, state, fitness, oos_sharpe, oos_log_growth,
                    pbo, dsr_pvalue, turnover, correlation_avg,
                    created_at, updated_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                values,
            )
        self._conn.commit()

    def list_all(self) -> list[AlphaRecord]:
        rows = self._conn.execute("SELECT * FROM alphas").fetchall()
        return [self._row_to_record(r) for r in rows]

    def clear_deployed_alphas(self) -> None:
        self._conn.execute("DELETE FROM deployed_alphas")
        self._conn.commit()

    def queue_candidate_expressions(
        self,
        expressions: list[str],
        *,
        source: str,
        fitness: float = 0.0,
        behavior_json: dict | None = None,
        created_at: float | None = None,
    ) -> int:
        return queue_candidate_expressions(
            self._conn,
            expressions,
            source=source,
            fitness=fitness,
            behavior_json=behavior_json,
            created_at=created_at,
        )

    def queue_candidates(self, seeds: list[CandidateSeed]) -> int:
        return queue_candidates(self._conn, seeds)

    def list_candidate_expressions(
        self,
        *,
        statuses: tuple[str, ...] | None = None,
    ) -> list[str]:
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            rows = self._conn.execute(
                f"SELECT expression FROM candidates WHERE status IN ({placeholders})",
                statuses,
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT expression FROM candidates"
            ).fetchall()
        return [row["expression"] for row in rows]

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
            self._record_values(record),
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
        state = AlphaState.canonical(state)
        rows = self._conn.execute(
            "SELECT * FROM alphas WHERE state = ? ORDER BY oos_sharpe DESC",
            (state,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_active(self) -> list[AlphaRecord]:
        return self.list_by_state(AlphaState.ACTIVE)

    def list_deployed_alphas(self) -> list[AlphaRecord]:
        rows = self._conn.execute(
            """
            SELECT a.*
            FROM deployed_alphas u
            JOIN alphas a ON a.alpha_id = u.alpha_id
            ORDER BY u.slot ASC
            """
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_deployed_alpha_entries(self) -> list[DeployedAlphaEntry]:
        rows = self._conn.execute(
            """
            SELECT alpha_id, slot, deployed_at, deployment_score, metadata
            FROM deployed_alphas
            ORDER BY slot ASC
            """
        ).fetchall()
        return [
            DeployedAlphaEntry(
                alpha_id=row["alpha_id"],
                slot=row["slot"],
                deployed_at=row["deployed_at"],
                deployment_score=row["deployment_score"],
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]

    def deployed_alpha_ids(self) -> list[str]:
        return [entry.alpha_id for entry in self.list_deployed_alpha_entries()]

    def count_deployed_alphas(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM deployed_alphas").fetchone()
        return row[0]

    def replace_deployed_alphas(
        self,
        alpha_ids: list[str],
        *,
        scores: dict[str, float] | None = None,
        metadata: dict[str, dict] | None = None,
        deployed_at: float | None = None,
    ) -> None:
        unique_ids = list(dict.fromkeys(alpha_ids))
        if unique_ids:
            placeholders = ", ".join("?" for _ in unique_ids)
            rows = self._conn.execute(
                f"SELECT alpha_id FROM alphas WHERE alpha_id IN ({placeholders})",
                unique_ids,
            ).fetchall()
            existing_ids = {row["alpha_id"] for row in rows}
            missing = [alpha_id for alpha_id in unique_ids if alpha_id not in existing_ids]
            if missing:
                raise KeyError(f"Unknown alpha ids for deployed alphas: {missing[:5]}")

        stamp = deployed_at or time.time()
        self._conn.execute("DELETE FROM deployed_alphas")
        if unique_ids:
            rows = []
            for slot, alpha_id in enumerate(unique_ids):
                rows.append(
                    (
                        alpha_id,
                        slot,
                        stamp,
                        float((scores or {}).get(alpha_id, 0.0)),
                        json.dumps((metadata or {}).get(alpha_id, {})),
                    )
                )
            self._conn.executemany(
                """
                INSERT INTO deployed_alphas
                    (alpha_id, slot, deployed_at, deployment_score, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        self._conn.commit()

    def update_state(self, alpha_id: str, new_state: str) -> None:
        new_state = AlphaState.canonical(new_state)
        self._conn.execute(
            "UPDATE alphas SET state = ?, updated_at = ? WHERE alpha_id = ?",
            (new_state, time.time(), alpha_id),
        )
        self._conn.commit()

    def bulk_update_states(self, alpha_ids: list[str], new_state: str) -> None:
        alpha_ids = list(dict.fromkeys(alpha_ids))
        if not alpha_ids:
            return
        new_state = AlphaState.canonical(new_state)
        stamp = time.time()
        self._conn.executemany(
            "UPDATE alphas SET state = ?, updated_at = ? WHERE alpha_id = ?",
            [(new_state, stamp, alpha_id) for alpha_id in alpha_ids],
        )
        self._conn.commit()

    def update_stake(self, alpha_id: str, stake: float) -> None:
        self._conn.execute(
            "UPDATE alphas SET stake = ?, updated_at = ? WHERE alpha_id = ?",
            (stake, time.time(), alpha_id),
        )
        self._conn.commit()

    def bulk_update_stakes(self, stakes: dict[str, float]) -> None:
        if not stakes:
            return
        stamp = time.time()
        self._conn.executemany(
            "UPDATE alphas SET stake = ?, updated_at = ? WHERE alpha_id = ?",
            [(s, stamp, aid) for aid, s in stakes.items()],
        )
        self._conn.commit()

    def top_by_stake(self, n: int = 30) -> list[AlphaRecord]:
        rows = self._conn.execute(
            "SELECT * FROM alphas WHERE stake > 0 ORDER BY stake DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

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
            state = AlphaState.canonical(state)
            row = self._conn.execute(
                "SELECT COUNT(*) FROM alphas WHERE state = ?", (state,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM alphas").fetchone()
        return row[0]

    _ORDER_COLUMN = {"sharpe": "oos_sharpe", "log_growth": "oos_log_growth"}

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
    def _record_values(record: AlphaRecord) -> tuple:
        now = time.time()
        if record.created_at == 0.0:
            record.created_at = now
        if record.updated_at == 0.0:
            record.updated_at = now
        state = AlphaState.canonical(record.state)
        return (
            record.alpha_id,
            record.expression,
            state,
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
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> AlphaRecord:
        keys = row.keys()
        return AlphaRecord(
            alpha_id=row["alpha_id"],
            expression=row["expression"],
            state=AlphaState.canonical(row["state"]),
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
            stake=row["stake"] if "stake" in keys else 0.0,
        )
