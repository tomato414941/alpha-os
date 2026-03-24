from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DATA_DIR


class HypothesisKind:
    DSL = "dsl"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    ML = "ml"
    EXTERNAL = "external"
    MANUAL = "manual"

    _CANONICAL = {
        DSL: DSL,
        TECHNICAL: TECHNICAL,
        FUNDAMENTAL: FUNDAMENTAL,
        ML: ML,
        EXTERNAL: EXTERNAL,
        MANUAL: MANUAL,
    }

    @classmethod
    def canonical(cls, kind: str) -> str:
        return cls._CANONICAL.get(kind, kind)


class HypothesisStatus:
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"

    _CANONICAL = {
        ACTIVE: ACTIVE,
        PAUSED: PAUSED,
        ARCHIVED: ARCHIVED,
    }

    @classmethod
    def canonical(cls, status: str) -> str:
        return cls._CANONICAL.get(status, status)


@dataclass
class HypothesisRecord:
    hypothesis_id: str
    kind: str
    definition: dict
    name: str = ""
    status: str = HypothesisStatus.ACTIVE
    stake: float = 0.0
    target_kind: str = "forward_residual_return"
    horizon: str = "20D2L"
    source: str = ""
    scope: dict = field(default_factory=lambda: {"universe": "core_universe_1000"})
    metadata: dict = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    _OOS_FITNESS_MAP = {"sharpe": "oos_sharpe", "log_growth": "oos_log_growth"}

    @property
    def alpha_id(self) -> str:
        """Compatibility alias for legacy runtime code."""
        return self.hypothesis_id

    @property
    def expression(self) -> str:
        """Return the DSL expression when this hypothesis is DSL-backed."""
        return str(self.definition.get("expression", ""))

    def oos_fitness(self, metric: str = "sharpe") -> float:
        key = self._OOS_FITNESS_MAP[metric]
        value = self.metadata.get(key, 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


@dataclass(frozen=True)
class HypothesisContribution:
    hypothesis_id: str
    date: str
    contribution: float
    created_at: float = 0.0


class HypothesisStore:
    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or DATA_DIR / "hypotheses.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id   TEXT PRIMARY KEY,
                kind            TEXT NOT NULL,
                name            TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL DEFAULT 'active',
                stake           REAL NOT NULL DEFAULT 0.0,
                target_kind     TEXT NOT NULL DEFAULT 'forward_residual_return',
                horizon         TEXT NOT NULL DEFAULT '20D2L',
                source          TEXT NOT NULL DEFAULT '',
                scope_json      TEXT NOT NULL DEFAULT '{"universe":"core_universe_1000"}',
                definition_json TEXT NOT NULL,
                metadata_json   TEXT NOT NULL DEFAULT '{}',
                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_hypotheses_status
            ON hypotheses(status)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_hypotheses_kind
            ON hypotheses(kind)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_hypotheses_stake
            ON hypotheses(stake DESC)
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hypothesis_contributions (
                hypothesis_id TEXT NOT NULL,
                date TEXT NOT NULL,
                contribution REAL NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (hypothesis_id, date)
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_hypothesis_contributions_date
            ON hypothesis_contributions(date)
            """
        )
        self._conn.commit()

    def register(self, record: HypothesisRecord) -> None:
        now = time.time()
        if record.created_at == 0.0:
            record.created_at = now
        record.updated_at = now
        self._conn.execute(
            """
            INSERT OR REPLACE INTO hypotheses
            (
                hypothesis_id,
                kind,
                name,
                status,
                stake,
                target_kind,
                horizon,
                source,
                scope_json,
                definition_json,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._record_values(record),
        )
        self._conn.commit()

    def get(self, hypothesis_id: str) -> HypothesisRecord | None:
        row = self._conn.execute(
            "SELECT * FROM hypotheses WHERE hypothesis_id = ?",
            (hypothesis_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_all(self) -> list[HypothesisRecord]:
        rows = self._conn.execute(
            "SELECT * FROM hypotheses ORDER BY updated_at DESC, hypothesis_id ASC"
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_active(self) -> list[HypothesisRecord]:
        return self.list_capital_backed()

    def list_capital_eligible(self) -> list[HypothesisRecord]:
        records = self.list_observation_active()
        return [
            record
            for record in records
            if bool(record.metadata.get("lifecycle_capital_eligible", record.stake > 0))
        ]

    def list_capital_backed(self, *, floor: float = 0.0) -> list[HypothesisRecord]:
        records = self.list_capital_eligible()
        return [record for record in records if float(record.stake) > floor]

    def list_live(self) -> list[HypothesisRecord]:
        return self.list_capital_backed()

    def top_by_stake(self, n: int = 30) -> list[HypothesisRecord]:
        records = self.list_capital_backed()
        return sorted(
            records,
            key=lambda record: (-float(record.stake), -float(record.updated_at)),
        )[:n]

    def list_active_sql_legacy(self) -> list[HypothesisRecord]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM hypotheses
            WHERE status = ? AND stake > 0
            ORDER BY stake DESC, updated_at DESC
            """,
            (HypothesisStatus.ACTIVE,),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_observation_active(self) -> list[HypothesisRecord]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM hypotheses
            WHERE status = ?
            ORDER BY stake DESC, updated_at DESC
            """,
            (HypothesisStatus.ACTIVE,),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_by_status(self, status: str) -> list[HypothesisRecord]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM hypotheses
            WHERE status = ?
            ORDER BY updated_at DESC, hypothesis_id ASC
            """,
            (HypothesisStatus.canonical(status),),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def count(self, *, status: str | None = None) -> int:
        if status is None:
            row = self._conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM hypotheses WHERE status = ?",
                (HypothesisStatus.canonical(status),),
            ).fetchone()
        return row[0]

    def update_status(self, hypothesis_id: str, status: str) -> None:
        self._conn.execute(
            """
            UPDATE hypotheses
            SET status = ?, updated_at = ?
            WHERE hypothesis_id = ?
            """,
            (HypothesisStatus.canonical(status), time.time(), hypothesis_id),
        )
        self._conn.commit()

    def update_stake(self, hypothesis_id: str, stake: float) -> None:
        self._conn.execute(
            """
            UPDATE hypotheses
            SET stake = ?, updated_at = ?
            WHERE hypothesis_id = ?
            """,
            (stake, time.time(), hypothesis_id),
        )
        self._conn.commit()

    def update_metadata(
        self,
        hypothesis_id: str,
        metadata: dict,
        *,
        merge: bool = True,
    ) -> None:
        row = self._conn.execute(
            "SELECT metadata_json FROM hypotheses WHERE hypothesis_id = ?",
            (hypothesis_id,),
        ).fetchone()
        if row is None:
            return
        if merge:
            current = json.loads(row["metadata_json"])
            current.update(metadata)
            metadata = current
        self._conn.execute(
            """
            UPDATE hypotheses
            SET metadata_json = ?, updated_at = ?
            WHERE hypothesis_id = ?
            """,
            (json.dumps(metadata), time.time(), hypothesis_id),
        )
        self._conn.commit()

    def record_contribution(
        self,
        hypothesis_id: str,
        *,
        date: str,
        contribution: float,
        created_at: float | None = None,
    ) -> None:
        stamp = created_at or time.time()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO hypothesis_contributions
            (hypothesis_id, date, contribution, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (hypothesis_id, date, float(contribution), stamp),
        )
        self._conn.commit()

    def list_contributions(
        self,
        hypothesis_id: str,
        *,
        limit: int | None = None,
    ) -> list[HypothesisContribution]:
        if limit is None:
            rows = self._conn.execute(
                """
                SELECT hypothesis_id, date, contribution, created_at
                FROM hypothesis_contributions
                WHERE hypothesis_id = ?
                ORDER BY date DESC
                """,
                (hypothesis_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT hypothesis_id, date, contribution, created_at
                FROM hypothesis_contributions
                WHERE hypothesis_id = ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (hypothesis_id, limit),
            ).fetchall()
        return [self._row_to_contribution(row) for row in rows]

    def contribution_history(
        self,
        hypothesis_id: str,
        *,
        limit: int | None = None,
    ) -> list[float]:
        return [
            row.contribution
            for row in self.list_contributions(hypothesis_id, limit=limit)
        ]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _record_values(record: HypothesisRecord) -> tuple:
        return (
            record.hypothesis_id,
            HypothesisKind.canonical(record.kind),
            record.name,
            HypothesisStatus.canonical(record.status),
            float(record.stake),
            record.target_kind,
            record.horizon,
            record.source,
            json.dumps(record.scope),
            json.dumps(record.definition),
            json.dumps(record.metadata),
            record.created_at,
            record.updated_at,
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> HypothesisRecord:
        return HypothesisRecord(
            hypothesis_id=row["hypothesis_id"],
            kind=HypothesisKind.canonical(row["kind"]),
            name=row["name"],
            status=HypothesisStatus.canonical(row["status"]),
            stake=row["stake"],
            target_kind=row["target_kind"],
            horizon=row["horizon"],
            source=row["source"],
            scope=json.loads(row["scope_json"]),
            definition=json.loads(row["definition_json"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_contribution(row: sqlite3.Row) -> HypothesisContribution:
        return HypothesisContribution(
            hypothesis_id=row["hypothesis_id"],
            date=row["date"],
            contribution=row["contribution"],
            created_at=row["created_at"],
        )
