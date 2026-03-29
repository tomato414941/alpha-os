from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import DEFAULT_ASSET, DEFAULT_TARGET
from .hypothesis_registry import find_hypothesis_definition
from .targets import TargetDefinition, find_target_definition, list_target_definitions
from .transition_policy import decide_operator_transition


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class HypothesisState:
    hypothesis_id: str
    asset: str
    target_id: str
    definition_json: str | None
    status: str
    prediction_count: int
    observation_count: int

    @property
    def definition(self) -> dict[str, Any] | None:
        if self.definition_json is None:
            return None
        return json.loads(self.definition_json)

    @property
    def target_definition(self) -> TargetDefinition | None:
        definition = self.definition
        if definition is None:
            return None
        target_document = definition.get("target_definition")
        if not isinstance(target_document, dict):
            return None
        return TargetDefinition.from_document(target_document)

    @property
    def kind(self) -> str | None:
        definition = self.definition
        if definition is None:
            return None
        value = definition.get("kind")
        return value if isinstance(value, str) else None

    @property
    def signal_name(self) -> str | None:
        definition = self.definition
        if definition is None:
            return None
        value = definition.get("signal_name")
        return value if isinstance(value, str) else None

    @property
    def lookback(self) -> int | None:
        definition = self.definition
        if definition is None:
            return None
        params = definition.get("params")
        if not isinstance(params, dict):
            return None
        value = params.get("lookback")
        return value if isinstance(value, int) else None

    @property
    def horizon_days(self) -> int | None:
        target_definition = self.target_definition
        if target_definition is None:
            return None
        return target_definition.horizon_days


@dataclass(frozen=True)
class EvaluationSnapshot:
    evaluation_id: str
    asset: str
    target_id: str
    hypothesis_id: str
    prediction_value: float
    observation_value: float
    signed_edge: float
    absolute_error: float
    input_source: str | None
    input_range_start: str | None
    input_range_end: str | None
    signal_name: str | None
    created_at: str


@dataclass(frozen=True)
class PredictionRecord:
    evaluation_id: str
    hypothesis_id: str
    asset: str
    target_id: str
    value: float
    recorded_at: str


@dataclass(frozen=True)
class ObservationRecord:
    evaluation_id: str
    asset: str
    target_id: str
    value: float
    recorded_at: str


@dataclass(frozen=True)
class TargetState:
    target_id: str
    definition_json: str

    @property
    def definition(self) -> TargetDefinition:
        return TargetDefinition.from_document(json.loads(self.definition_json))


@dataclass(frozen=True)
class HypothesisMetricState:
    hypothesis_id: str
    corr: float
    mmc: float | None
    mmc_baseline_type: str | None
    mmc_peer_count: int
    sample_count: int
    mmc_sample_count: int
    window_size: int
    start_evaluation_id: str | None
    end_evaluation_id: str | None
    updated_at: str


@dataclass(frozen=True)
class MetaPredictionState:
    evaluation_id: str
    asset: str
    target_id: str
    aggregation_kind: str
    value: float
    contributor_count: int
    details_json: str | None
    created_at: str
    updated_at: str

    @property
    def details(self) -> dict[str, Any] | None:
        if self.details_json is None:
            return None
        return json.loads(self.details_json)


@dataclass(frozen=True)
class MetaPredictionMetricState:
    aggregation_kind: str
    asset: str
    target_id: str
    corr: float
    sample_count: int
    window_size: int
    start_evaluation_id: str | None
    end_evaluation_id: str | None
    updated_at: str


def _row_to_hypothesis(row: sqlite3.Row | None) -> HypothesisState | None:
    if row is None:
        return None
    return HypothesisState(
        hypothesis_id=str(row["hypothesis_id"]),
        asset=str(row["asset"]),
        target_id=str(row["target_id"]),
        definition_json=None
        if row["definition_json"] is None
        else str(row["definition_json"]),
        status=str(row["status"]),
        prediction_count=int(row["prediction_count"]),
        observation_count=int(row["observation_count"]),
    )


def _row_to_snapshot(row: sqlite3.Row | None) -> EvaluationSnapshot | None:
    if row is None:
        return None
    return EvaluationSnapshot(
        evaluation_id=str(row["evaluation_id"]),
        asset=str(row["asset"]),
        target_id=str(row["target_id"]),
        hypothesis_id=str(row["hypothesis_id"]),
        prediction_value=float(row["prediction_value"]),
        observation_value=float(row["observation_value"]),
        signed_edge=float(row["signed_edge"]),
        absolute_error=float(row["absolute_error"]),
        input_source=None if row["input_source"] is None else str(row["input_source"]),
        input_range_start=None
        if row["input_range_start"] is None
        else str(row["input_range_start"]),
        input_range_end=None
        if row["input_range_end"] is None
        else str(row["input_range_end"]),
        signal_name=None if row["signal_name"] is None else str(row["signal_name"]),
        created_at=str(row["created_at"]),
    )


def _row_to_prediction(row: sqlite3.Row | None) -> PredictionRecord | None:
    if row is None:
        return None
    return PredictionRecord(
        evaluation_id=str(row["evaluation_id"]),
        hypothesis_id=str(row["hypothesis_id"]),
        asset=str(row["asset"]),
        target_id=str(row["target_id"]),
        value=float(row["value"]),
        recorded_at=str(row["recorded_at"]),
    )


def _row_to_observation(row: sqlite3.Row | None) -> ObservationRecord | None:
    if row is None:
        return None
    return ObservationRecord(
        evaluation_id=str(row["evaluation_id"]),
        asset=str(row["asset"]),
        target_id=str(row["target_id"]),
        value=float(row["value"]),
        recorded_at=str(row["recorded_at"]),
    )


def _row_to_hypothesis_metric(row: sqlite3.Row | None) -> HypothesisMetricState | None:
    if row is None:
        return None
    return HypothesisMetricState(
        hypothesis_id=str(row["hypothesis_id"]),
        corr=float(row["corr"]),
        mmc=None if row["mmc"] is None else float(row["mmc"]),
        mmc_baseline_type=None
        if row["mmc_baseline_type"] is None
        else str(row["mmc_baseline_type"]),
        mmc_peer_count=int(row["mmc_peer_count"]),
        sample_count=int(row["sample_count"]),
        mmc_sample_count=int(row["mmc_sample_count"]),
        window_size=int(row["window_size"]),
        start_evaluation_id=None
        if row["start_evaluation_id"] is None
        else str(row["start_evaluation_id"]),
        end_evaluation_id=None if row["end_evaluation_id"] is None else str(row["end_evaluation_id"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_target(row: sqlite3.Row | None) -> TargetState | None:
    if row is None:
        return None
    return TargetState(
        target_id=str(row["target_id"]),
        definition_json=str(row["definition_json"]),
    )


def _row_to_meta_prediction(row: sqlite3.Row | None) -> MetaPredictionState | None:
    if row is None:
        return None
    return MetaPredictionState(
        evaluation_id=str(row["evaluation_id"]),
        asset=str(row["asset"]),
        target_id=str(row["target_id"]),
        aggregation_kind=str(row["aggregation_kind"]),
        value=float(row["value"]),
        contributor_count=int(row["contributor_count"]),
        details_json=None if row["details_json"] is None else str(row["details_json"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_meta_prediction_metric(row: sqlite3.Row | None) -> MetaPredictionMetricState | None:
    if row is None:
        return None
    return MetaPredictionMetricState(
        aggregation_kind=str(row["aggregation_kind"]),
        asset=str(row["asset"]),
        target_id=str(row["target_id"]),
        corr=float(row["corr"]),
        sample_count=int(row["sample_count"]),
        window_size=int(row["window_size"]),
        start_evaluation_id=None
        if row["start_evaluation_id"] is None
        else str(row["start_evaluation_id"]),
        end_evaluation_id=None
        if row["end_evaluation_id"] is None
        else str(row["end_evaluation_id"]),
        updated_at=str(row["updated_at"]),
    )


class EvaluationStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS targets (
                target_id TEXT PRIMARY KEY,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                definition_json TEXT,
                status TEXT NOT NULL,
                prediction_count INTEGER NOT NULL,
                observation_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                evaluation_id TEXT NOT NULL,
                hypothesis_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                value REAL NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (evaluation_id, hypothesis_id)
            );

            CREATE TABLE IF NOT EXISTS observations (
                evaluation_id TEXT PRIMARY KEY,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                value REAL NOT NULL,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hypothesis_metrics (
                hypothesis_id TEXT PRIMARY KEY,
                corr REAL NOT NULL,
                mmc REAL,
                mmc_baseline_type TEXT,
                mmc_peer_count INTEGER NOT NULL,
                sample_count INTEGER NOT NULL,
                mmc_sample_count INTEGER NOT NULL,
                window_size INTEGER NOT NULL,
                start_evaluation_id TEXT,
                end_evaluation_id TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_snapshots (
                evaluation_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                hypothesis_id TEXT NOT NULL,
                prediction_value REAL NOT NULL,
                observation_value REAL NOT NULL,
                signed_edge REAL NOT NULL,
                absolute_error REAL NOT NULL,
                input_source TEXT,
                input_range_start TEXT,
                input_range_end TEXT,
                signal_name TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (evaluation_id, hypothesis_id)
            );

            CREATE TABLE IF NOT EXISTS meta_predictions (
                evaluation_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                aggregation_kind TEXT NOT NULL,
                value REAL NOT NULL,
                contributor_count INTEGER NOT NULL,
                details_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (evaluation_id, aggregation_kind)
            );

            CREATE TABLE IF NOT EXISTS meta_prediction_metrics (
                aggregation_kind TEXT NOT NULL,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                corr REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                window_size INTEGER NOT NULL,
                start_evaluation_id TEXT,
                end_evaluation_id TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (aggregation_kind, asset, target_id)
            );
            """
        )
        self._seed_builtin_targets()
        self.conn.commit()

    def _seed_builtin_targets(self) -> None:
        timestamp = _utc_now()
        for definition in list_target_definitions():
            self.conn.execute(
                """
                INSERT INTO targets (target_id, definition_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(target_id) DO NOTHING
                """,
                (
                    definition.target_id,
                    json.dumps(definition.to_document(), sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )

    def get_target(self, target_id: str) -> TargetState | None:
        row = self.conn.execute(
            """
            SELECT target_id, definition_json
            FROM targets
            WHERE target_id = ?
            """,
            (target_id,),
        ).fetchone()
        return _row_to_target(row)

    def register_target(
        self,
        target_id: str,
        *,
        definition: TargetDefinition | None = None,
        recorded_at: str | None = None,
    ) -> TargetState:
        self.ensure_schema()
        existing = self.get_target(target_id)
        if existing is not None:
            return existing

        resolved_definition = definition or find_target_definition(target_id)
        if resolved_definition is None:
            raise ValueError(f"target definition must exist before use: {target_id}")
        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO targets (target_id, definition_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    resolved_definition.target_id,
                    json.dumps(resolved_definition.to_document(), sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )
        target_state = self.get_target(target_id)
        assert target_state is not None
        return target_state

    def get_hypothesis(self, hypothesis_id: str) -> HypothesisState | None:
        row = self.conn.execute(
            """
            SELECT hypothesis_id, asset, target_id,
                   definition_json, status,
                   prediction_count, observation_count
            FROM hypotheses
            WHERE hypothesis_id = ?
            """,
            (hypothesis_id,),
        ).fetchone()
        return _row_to_hypothesis(row)

    def list_hypotheses(
        self,
        *,
        asset: str = DEFAULT_ASSET,
        target_id: str | None = DEFAULT_TARGET,
    ) -> list[HypothesisState]:
        if target_id is None:
            rows = self.conn.execute(
                """
                SELECT hypothesis_id, asset, target_id,
                       definition_json, status,
                       prediction_count, observation_count
                FROM hypotheses
                WHERE asset = ?
                ORDER BY target_id ASC, observation_count DESC, prediction_count DESC, hypothesis_id ASC
                """,
                (asset,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT hypothesis_id, asset, target_id,
                       definition_json, status,
                       prediction_count, observation_count
                FROM hypotheses
                WHERE asset = ? AND target_id = ?
                ORDER BY observation_count DESC, prediction_count DESC, hypothesis_id ASC
                """,
                (asset, target_id),
            ).fetchall()
        return [_row_to_hypothesis(row) for row in rows if row is not None]

    def get_hypothesis_metric(self, hypothesis_id: str) -> HypothesisMetricState | None:
        row = self.conn.execute(
            """
            SELECT hypothesis_id, corr, mmc, sample_count, window_size,
                   mmc_baseline_type, mmc_peer_count, mmc_sample_count,
                   start_evaluation_id, end_evaluation_id, updated_at
            FROM hypothesis_metrics
            WHERE hypothesis_id = ?
            """,
            (hypothesis_id,),
        ).fetchone()
        return _row_to_hypothesis_metric(row)

    def list_hypothesis_metrics(
        self,
        *,
        hypothesis_ids: list[str] | None = None,
    ) -> list[HypothesisMetricState]:
        if not hypothesis_ids:
            rows = self.conn.execute(
                """
                SELECT hypothesis_id, corr, mmc, mmc_baseline_type, mmc_peer_count,
                       sample_count, mmc_sample_count, window_size,
                       start_evaluation_id, end_evaluation_id, updated_at
                FROM hypothesis_metrics
                ORDER BY corr DESC, mmc DESC, hypothesis_id ASC
                """
            ).fetchall()
            return [_row_to_hypothesis_metric(row) for row in rows if row is not None]

        placeholders = ", ".join("?" for _ in hypothesis_ids)
        rows = self.conn.execute(
            f"""
            SELECT hypothesis_id, corr, mmc, mmc_baseline_type, mmc_peer_count,
                   sample_count, mmc_sample_count, window_size,
                   start_evaluation_id, end_evaluation_id, updated_at
            FROM hypothesis_metrics
            WHERE hypothesis_id IN ({placeholders})
            ORDER BY corr DESC, mmc DESC, hypothesis_id ASC
            """,
            tuple(hypothesis_ids),
        ).fetchall()
        return [_row_to_hypothesis_metric(row) for row in rows if row is not None]

    def upsert_meta_prediction(
        self,
        *,
        evaluation_id: str,
        asset: str,
        target_id: str,
        aggregation_kind: str,
        value: float,
        contributor_count: int,
        details_json: str | None,
        recorded_at: str | None = None,
    ) -> None:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO meta_predictions (
                    evaluation_id, asset, target_id, aggregation_kind, value,
                    contributor_count, details_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_id, aggregation_kind) DO UPDATE SET
                    asset = excluded.asset,
                    target_id = excluded.target_id,
                    value = excluded.value,
                    contributor_count = excluded.contributor_count,
                    details_json = excluded.details_json,
                    updated_at = excluded.updated_at
                """,
                (
                    evaluation_id,
                    asset,
                    target_id,
                    aggregation_kind,
                    float(value),
                    int(contributor_count),
                    details_json,
                    timestamp,
                    timestamp,
                ),
            )

    def list_meta_predictions(
        self,
        *,
        asset: str = DEFAULT_ASSET,
        target_id: str | None = None,
        aggregation_kind: str | None = None,
        limit: int = 20,
    ) -> list[MetaPredictionState]:
        filters = ["asset = ?"]
        params: list[Any] = [asset]
        if target_id is not None:
            filters.append("target_id = ?")
            params.append(target_id)
        if aggregation_kind is not None:
            filters.append("aggregation_kind = ?")
            params.append(aggregation_kind)
        params.append(max(int(limit), 1))
        rows = self.conn.execute(
            f"""
            SELECT evaluation_id, asset, target_id, aggregation_kind, value,
                   contributor_count, details_json, created_at, updated_at
            FROM meta_predictions
            WHERE {' AND '.join(filters)}
            ORDER BY updated_at DESC, evaluation_id DESC, aggregation_kind ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_meta_prediction(row) for row in rows if row is not None]

    def upsert_meta_prediction_metric(
        self,
        *,
        aggregation_kind: str,
        asset: str,
        target_id: str,
        corr: float,
        sample_count: int,
        window_size: int,
        start_evaluation_id: str | None,
        end_evaluation_id: str | None,
        recorded_at: str | None = None,
    ) -> None:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO meta_prediction_metrics (
                    aggregation_kind, asset, target_id, corr, sample_count,
                    window_size, start_evaluation_id, end_evaluation_id, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(aggregation_kind, asset, target_id) DO UPDATE SET
                    corr = excluded.corr,
                    sample_count = excluded.sample_count,
                    window_size = excluded.window_size,
                    start_evaluation_id = excluded.start_evaluation_id,
                    end_evaluation_id = excluded.end_evaluation_id,
                    updated_at = excluded.updated_at
                """,
                (
                    aggregation_kind,
                    asset,
                    target_id,
                    float(corr),
                    int(sample_count),
                    int(window_size),
                    start_evaluation_id,
                    end_evaluation_id,
                    timestamp,
                ),
            )

    def list_meta_prediction_metrics(
        self,
        *,
        asset: str = DEFAULT_ASSET,
        target_id: str | None = None,
    ) -> list[MetaPredictionMetricState]:
        filters = ["asset = ?"]
        params: list[Any] = [asset]
        if target_id is not None:
            filters.append("target_id = ?")
            params.append(target_id)
        rows = self.conn.execute(
            f"""
            SELECT aggregation_kind, asset, target_id, corr, sample_count,
                   window_size, start_evaluation_id, end_evaluation_id, updated_at
            FROM meta_prediction_metrics
            WHERE {' AND '.join(filters)}
            ORDER BY target_id ASC, corr DESC, aggregation_kind ASC
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_meta_prediction_metric(row) for row in rows if row is not None]

    def set_hypothesis_status(
        self,
        hypothesis_id: str,
        *,
        action: str,
        recorded_at: str | None = None,
    ) -> HypothesisState:
        self.ensure_schema()
        hypothesis = self.get_hypothesis(hypothesis_id)
        if hypothesis is None:
            raise ValueError(f"hypothesis does not exist: {hypothesis_id}")
        decision = decide_operator_transition(
            current_status=hypothesis.status,
            action=action,
        )

        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                UPDATE hypotheses
                SET status = ?, updated_at = ?
                WHERE hypothesis_id = ?
                """,
                (decision.next_status, timestamp, hypothesis_id),
            )

        updated = self.get_hypothesis(hypothesis_id)
        assert updated is not None
        return updated

    def register_hypothesis(
        self,
        hypothesis_id: str,
        *,
        recorded_at: str | None = None,
        asset: str = DEFAULT_ASSET,
        target_id: str = DEFAULT_TARGET,
    ) -> tuple[HypothesisState, bool]:
        self.ensure_schema()
        existing = self.get_hypothesis(hypothesis_id)
        if existing is not None:
            return existing, False

        timestamp = recorded_at or _utc_now()
        definition = find_hypothesis_definition(hypothesis_id)
        if (
            definition is not None
            and target_id != DEFAULT_TARGET
            and target_id != definition.target_id
        ):
            raise ValueError(
                "built-in hypothesis target does not match provided target: "
                f"{target_id} != {definition.target_id}"
            )
        definition_json = (
            None
            if definition is None
            else json.dumps(definition.to_document(), sort_keys=True)
        )
        resolved_asset = asset if definition is None else definition.asset
        resolved_target_id = target_id if definition is None else definition.target_id
        self.register_target(
            resolved_target_id,
            definition=None if definition is None else definition.target,
            recorded_at=timestamp,
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO hypotheses (
                    hypothesis_id, asset, target_id, definition_json, status,
                    prediction_count, observation_count, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 'active', 0, 0, ?, ?)
                """,
                (
                    hypothesis_id,
                    resolved_asset,
                    resolved_target_id,
                    definition_json,
                    timestamp,
                    timestamp,
                ),
            )

        hypothesis = self.get_hypothesis(hypothesis_id)
        assert hypothesis is not None
        return hypothesis, True

    def get_evaluation_snapshot(
        self,
        evaluation_id: str,
        hypothesis_id: str,
    ) -> EvaluationSnapshot | None:
        row = self.conn.execute(
            """
            SELECT evaluation_id, asset, target_id, hypothesis_id, prediction_value,
                   observation_value, signed_edge, absolute_error, input_source,
                   input_range_start, input_range_end, signal_name, created_at
            FROM evaluation_snapshots
            WHERE evaluation_id = ? AND hypothesis_id = ?
            """,
            (evaluation_id, hypothesis_id),
        ).fetchone()
        return _row_to_snapshot(row)

    def get_prediction(
        self,
        evaluation_id: str,
        hypothesis_id: str,
    ) -> PredictionRecord | None:
        row = self.conn.execute(
            """
            SELECT evaluation_id, hypothesis_id, asset, target_id, value, recorded_at
            FROM predictions
            WHERE evaluation_id = ? AND hypothesis_id = ?
            """,
            (evaluation_id, hypothesis_id),
        ).fetchone()
        return _row_to_prediction(row)

    def get_observation(self, evaluation_id: str) -> ObservationRecord | None:
        row = self.conn.execute(
            """
            SELECT evaluation_id, asset, target_id, value, recorded_at
            FROM observations
            WHERE evaluation_id = ?
            """,
            (evaluation_id,),
        ).fetchone()
        return _row_to_observation(row)

    def list_evaluation_snapshots(self, *, limit: int = 20) -> list[EvaluationSnapshot]:
        rows = self.conn.execute(
            """
            SELECT evaluation_id, asset, target_id, hypothesis_id, prediction_value,
                   observation_value, signed_edge, absolute_error, input_source,
                   input_range_start, input_range_end, signal_name, created_at
            FROM evaluation_snapshots
            ORDER BY created_at DESC, evaluation_id DESC, hypothesis_id DESC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [_row_to_snapshot(row) for row in rows]

    def record_prediction(
        self,
        *,
        evaluation_id: str,
        hypothesis_id: str,
        prediction_value: float,
        recorded_at: str | None = None,
        asset: str = DEFAULT_ASSET,
        target_id: str = DEFAULT_TARGET,
    ) -> tuple[PredictionRecord, bool]:
        self.ensure_schema()
        hypothesis = self.get_hypothesis(hypothesis_id)
        if hypothesis is None:
            raise ValueError(f"hypothesis must exist before recording predictions: {hypothesis_id}")
        if hypothesis.status != "active":
            raise ValueError(
                f"prediction cannot be recorded while hypothesis is {hypothesis.status}: "
                f"{hypothesis_id}"
            )
        if hypothesis.asset != asset:
            raise ValueError(
                f"prediction asset does not match hypothesis asset: {asset} != {hypothesis.asset}"
            )
        if hypothesis.target_id != target_id:
            raise ValueError(
                "prediction target does not match hypothesis target: "
                f"{target_id} != {hypothesis.target_id}"
            )
        self.register_target(target_id, recorded_at=recorded_at)

        existing = self.get_prediction(evaluation_id, hypothesis_id)
        if existing is not None:
            if existing.value != float(prediction_value):
                raise ValueError(
                    "prediction already exists for this evaluation_id and hypothesis_id with a "
                    "different value"
                )
            if existing.asset != asset or existing.target_id != target_id:
                raise ValueError(
                    "prediction already exists for this evaluation with different asset/target"
                )
            return existing, False

        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO predictions (
                    evaluation_id, hypothesis_id, asset, target_id, value, recorded_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    hypothesis_id,
                    asset,
                    target_id,
                    float(prediction_value),
                    timestamp,
                ),
            )

        prediction = self.get_prediction(evaluation_id, hypothesis_id)
        assert prediction is not None
        return prediction, True

    def finalize_observation(
        self,
        *,
        evaluation_id: str,
        observation_value: float,
        recorded_at: str | None = None,
        asset: str = DEFAULT_ASSET,
        target_id: str = DEFAULT_TARGET,
    ) -> tuple[ObservationRecord, bool]:
        self.ensure_schema()
        self.register_target(target_id, recorded_at=recorded_at)
        existing = self.get_observation(evaluation_id)
        if existing is not None:
            if existing.value != float(observation_value):
                raise ValueError(
                    "observation already exists for this evaluation_id with a different value"
                )
            if existing.asset != asset or existing.target_id != target_id:
                raise ValueError(
                    "observation already exists for this evaluation with different asset/target"
                )
            return existing, False

        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO observations (evaluation_id, asset, target_id, value, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (evaluation_id, asset, target_id, float(observation_value), timestamp),
            )

        observation = self.get_observation(evaluation_id)
        assert observation is not None
        return observation, True
