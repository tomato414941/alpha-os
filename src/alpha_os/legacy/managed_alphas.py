"""Managed alpha store backed by SQLite."""
# TODO: Legacy managed/deployed registry substrate. Keep this outside the
# hypotheses-first runtime mainline until remaining replay/simulator paths are
# retired or redesigned around hypotheses.db.
from __future__ import annotations

import sqlite3
from pathlib import Path

from ..config import DATA_DIR
from .alpha_registry import (
    bulk_update_states,
    bulk_update_stakes,
    count,
    ensure_alpha_tables,
    get,
    list_all,
    list_by_state,
    register,
    replace_all,
    row_to_record,
    top,
    top_by_stake,
    update_metrics,
    update_stake,
    update_state,
)
from .admission_queue import (
    CandidateSeed,
    list_candidate_expressions,
    queue_candidate_expressions,
    queue_candidates,
)
from .deployed_registry import (
    clear_deployed_alphas,
    count_deployed_alphas,
    list_deployed_alpha_entries,
    list_deployed_alphas,
    replace_deployed_alphas,
)
from .registry_types import AlphaRecord, AlphaState, DeployedAlphaEntry


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
        ensure_alpha_tables(self._conn)
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
        self._conn.commit()

    def replace_all(self, records: list[AlphaRecord]) -> None:
        self._conn.execute("DELETE FROM deployed_alphas")
        replace_all(self._conn, records)

    def list_all(self) -> list[AlphaRecord]:
        return list_all(self._conn)

    def clear_deployed_alphas(self) -> None:
        clear_deployed_alphas(self._conn)

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
        return list_candidate_expressions(self._conn, statuses=statuses)

    def register(self, record: AlphaRecord) -> None:
        register(self._conn, record)

    def get(self, alpha_id: str) -> AlphaRecord | None:
        return get(self._conn, alpha_id)

    def list_by_state(self, state: str) -> list[AlphaRecord]:
        return list_by_state(self._conn, state)

    def list_active(self) -> list[AlphaRecord]:
        return self.list_by_state(AlphaState.ACTIVE)

    def list_deployed_alphas(self) -> list[AlphaRecord]:
        return list_deployed_alphas(self._conn, row_to_record=row_to_record)

    def list_deployed_alpha_entries(self) -> list[DeployedAlphaEntry]:
        return list_deployed_alpha_entries(self._conn)

    def deployed_alpha_ids(self) -> list[str]:
        return [entry.alpha_id for entry in self.list_deployed_alpha_entries()]

    def count_deployed_alphas(self) -> int:
        return count_deployed_alphas(self._conn)

    def replace_deployed_alphas(
        self,
        alpha_ids: list[str],
        *,
        scores: dict[str, float] | None = None,
        metadata: dict[str, dict] | None = None,
        deployed_at: float | None = None,
    ) -> None:
        replace_deployed_alphas(
            self._conn,
            alpha_ids,
            scores=scores,
            metadata=metadata,
            deployed_at=deployed_at,
        )

    def update_state(self, alpha_id: str, new_state: str) -> None:
        update_state(self._conn, alpha_id, new_state)

    def bulk_update_states(self, alpha_ids: list[str], new_state: str) -> None:
        bulk_update_states(self._conn, alpha_ids, new_state)

    def update_stake(self, alpha_id: str, stake: float) -> None:
        update_stake(self._conn, alpha_id, stake)

    def bulk_update_stakes(self, stakes: dict[str, float]) -> None:
        bulk_update_stakes(self._conn, stakes)

    def top_by_stake(self, n: int = 30) -> list[AlphaRecord]:
        return top_by_stake(self._conn, n)

    def update_metrics(
        self, alpha_id: str, oos_sharpe: float | None = None,
        pbo: float | None = None, dsr_pvalue: float | None = None,
    ) -> None:
        update_metrics(
            self._conn,
            alpha_id,
            oos_sharpe=oos_sharpe,
            pbo=pbo,
            dsr_pvalue=dsr_pvalue,
        )

    def count(self, state: str | None = None) -> int:
        return count(self._conn, state)

    def top(self, n: int = 10, state: str | None = None, metric: str = "sharpe") -> list[AlphaRecord]:
        return top(self._conn, n=n, state=state, metric=metric)

    def close(self) -> None:
        self._conn.close()
