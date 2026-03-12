"""Alpha Registry — SQLite-backed metadata store for alpha lifecycle."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DATA_DIR


class AlphaState:
    CANDIDATE = "candidate"
    ACTIVE = "active"
    DORMANT = "dormant"
    REJECTED = "rejected"

    # Backward-compatible aliases for migrated runtime state names.
    BORN = CANDIDATE
    PROBATION = ACTIVE

    _CANONICAL = {
        CANDIDATE: CANDIDATE,
        ACTIVE: ACTIVE,
        DORMANT: DORMANT,
        REJECTED: REJECTED,
        "born": CANDIDATE,
        "probation": ACTIVE,
    }

    @classmethod
    def canonical(cls, state: str) -> str:
        return cls._CANONICAL.get(state, state)

    @classmethod
    def trading_states(cls) -> tuple[str, ...]:
        return (cls.ACTIVE,)

    @classmethod
    def runtime_states(cls) -> tuple[str, ...]:
        return (cls.ACTIVE, cls.DORMANT)


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
                state TEXT NOT NULL DEFAULT 'candidate',
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
        # Pipeline v2: candidate queue for evo daemon → admission flow
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
        # Pipeline v2: pre-computed diversity scores from admission daemon
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS diversity_cache (
                alpha_id TEXT PRIMARY KEY,
                diversity_score REAL NOT NULL,
                computed_at REAL NOT NULL,
                n_alphas_compared INTEGER NOT NULL
            )
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
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_oos_log_growth
            ON alphas(oos_log_growth DESC)
        """)
        self._migrate_deployed_alphas()
        self._migrate_states()
        self._conn.commit()

    def _migrate_deployed_alphas(self) -> None:
        row = self._conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'trading_universe'
            """
        ).fetchone()
        if row is None:
            return
        deployed_count = self._conn.execute(
            "SELECT COUNT(*) FROM deployed_alphas"
        ).fetchone()[0]
        if deployed_count > 0:
            return
        self._conn.execute(
            """
            INSERT INTO deployed_alphas (
                alpha_id, slot, deployed_at, deployment_score, metadata
            )
            SELECT alpha_id, slot, deployed_at, deployment_score, metadata
            FROM trading_universe
            """
        )

    def _migrate_states(self) -> None:
        self._conn.execute(
            "UPDATE alphas SET state = ? WHERE state = ?",
            (AlphaState.CANDIDATE, "born"),
        )
        self._conn.execute(
            "UPDATE alphas SET state = ? WHERE state = ?",
            (AlphaState.ACTIVE, "probation"),
        )

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

    def clear_diversity_cache(self) -> None:
        self._conn.execute("DELETE FROM diversity_cache")
        self._conn.commit()

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
        if not expressions:
            return 0

        stamp = created_at or time.time()
        payload: dict[str, tuple[str, str, float, str, str, float]] = {}
        behavior = json.dumps(behavior_json or {})
        for expression in expressions:
            digest = hashlib.md5(
                f"{source}:{expression}".encode(),
                usedforsecurity=False,
            ).hexdigest()[:16]
            candidate_id = f"{source}_{digest}"
            payload[candidate_id] = (
                candidate_id,
                expression,
                float(fitness),
                "pending",
                behavior,
                stamp,
            )

        rows = list(payload.values())
        placeholders = ", ".join("?" for _ in rows)
        existing = set()
        if rows:
            existing_rows = self._conn.execute(
                f"SELECT candidate_id FROM candidates WHERE candidate_id IN ({placeholders})",
                [row[0] for row in rows],
            ).fetchall()
            existing = {row["candidate_id"] for row in existing_rows}

        self._conn.executemany(
            """INSERT OR IGNORE INTO candidates
               (candidate_id, expression, fitness, status, behavior_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()
        return len(rows) - len(existing)

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

    def top_trading(self, n: int = 30, metric: str = "sharpe") -> list[AlphaRecord]:
        """Top-N alphas by fitness metric from active state (for trading)."""
        col = self._ORDER_COLUMN[metric]
        placeholders = ", ".join("?" for _ in AlphaState.trading_states())
        rows = self._conn.execute(
            f"SELECT * FROM alphas WHERE state IN ({placeholders}) "
            f"ORDER BY {col} DESC LIMIT ?",
            (*AlphaState.trading_states(), n),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def bottom_trading(self, n: int = 1, metric: str = "sharpe") -> list[AlphaRecord]:
        """Bottom-N active alphas by fitness metric."""
        col = self._ORDER_COLUMN[metric]
        placeholders = ", ".join("?" for _ in AlphaState.trading_states())
        rows = self._conn.execute(
            f"SELECT * FROM alphas WHERE state IN ({placeholders}) "
            f"ORDER BY {col} ASC LIMIT ?",
            (*AlphaState.trading_states(), n),
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
        )
