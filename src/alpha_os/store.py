from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .config import DEFAULT_ASSET, DEFAULT_TARGET
from .hypothesis_registry import find_hypothesis_definition
from .scoring import DEFAULT_METRIC_WINDOW, compute_hypothesis_metrics
from .transition_policy import decide_operator_transition


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class HypothesisState:
    hypothesis_id: str
    asset: str
    target: str
    kind: str | None
    signal_name: str | None
    lookback: int | None
    status: str
    prediction_count: int
    observation_count: int


@dataclass(frozen=True)
class EvaluationSnapshot:
    evaluation_id: str
    asset: str
    target: str
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
    target: str
    value: float
    recorded_at: str


@dataclass(frozen=True)
class ObservationRecord:
    evaluation_id: str
    asset: str
    target: str
    value: float
    recorded_at: str


@dataclass(frozen=True)
class HypothesisMetricState:
    hypothesis_id: str
    corr: float
    mmc: float
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
        target=str(row["target"]),
        kind=None if row["kind"] is None else str(row["kind"]),
        signal_name=None if row["signal_name"] is None else str(row["signal_name"]),
        lookback=None if row["lookback"] is None else int(row["lookback"]),
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
        target=str(row["target"]),
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
        target=str(row["target"]),
        value=float(row["value"]),
        recorded_at=str(row["recorded_at"]),
    )


def _row_to_observation(row: sqlite3.Row | None) -> ObservationRecord | None:
    if row is None:
        return None
    return ObservationRecord(
        evaluation_id=str(row["evaluation_id"]),
        asset=str(row["asset"]),
        target=str(row["target"]),
        value=float(row["value"]),
        recorded_at=str(row["recorded_at"]),
    )


def _row_to_hypothesis_metric(row: sqlite3.Row | None) -> HypothesisMetricState | None:
    if row is None:
        return None
    return HypothesisMetricState(
        hypothesis_id=str(row["hypothesis_id"]),
        corr=float(row["corr"]),
        mmc=float(row["mmc"]),
        sample_count=int(row["sample_count"]),
        window_size=int(row["window_size"]),
        start_evaluation_id=None
        if row["start_evaluation_id"] is None
        else str(row["start_evaluation_id"]),
        end_evaluation_id=None if row["end_evaluation_id"] is None else str(row["end_evaluation_id"]),
        updated_at=str(row["updated_at"]),
    )


class V1Store:
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
            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                asset TEXT NOT NULL,
                target TEXT NOT NULL,
                kind TEXT,
                signal_name TEXT,
                lookback INTEGER,
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
                target TEXT NOT NULL,
                value REAL NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (evaluation_id, hypothesis_id)
            );

            CREATE TABLE IF NOT EXISTS observations (
                evaluation_id TEXT PRIMARY KEY,
                asset TEXT NOT NULL,
                target TEXT NOT NULL,
                value REAL NOT NULL,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hypothesis_metrics (
                hypothesis_id TEXT PRIMARY KEY,
                corr REAL NOT NULL,
                mmc REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                window_size INTEGER NOT NULL,
                start_evaluation_id TEXT,
                end_evaluation_id TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_snapshots (
                evaluation_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target TEXT NOT NULL,
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
            """
        )
        self.conn.commit()

    def get_hypothesis(self, hypothesis_id: str) -> HypothesisState | None:
        row = self.conn.execute(
            """
            SELECT hypothesis_id, asset, target,
                   kind, signal_name, lookback, status,
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
        target: str = DEFAULT_TARGET,
    ) -> list[HypothesisState]:
        rows = self.conn.execute(
            """
            SELECT hypothesis_id, asset, target,
                   kind, signal_name, lookback, status,
                   prediction_count, observation_count
            FROM hypotheses
            WHERE asset = ? AND target = ?
            ORDER BY observation_count DESC, prediction_count DESC, hypothesis_id ASC
            """,
            (asset, target),
        ).fetchall()
        return [_row_to_hypothesis(row) for row in rows if row is not None]

    def get_hypothesis_metric(self, hypothesis_id: str) -> HypothesisMetricState | None:
        row = self.conn.execute(
            """
            SELECT hypothesis_id, corr, mmc, sample_count, window_size,
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
                SELECT hypothesis_id, corr, mmc, sample_count, window_size,
                       start_evaluation_id, end_evaluation_id, updated_at
                FROM hypothesis_metrics
                ORDER BY corr DESC, mmc DESC, hypothesis_id ASC
                """
            ).fetchall()
            return [_row_to_hypothesis_metric(row) for row in rows if row is not None]

        placeholders = ", ".join("?" for _ in hypothesis_ids)
        rows = self.conn.execute(
            f"""
            SELECT hypothesis_id, corr, mmc, sample_count, window_size,
                   start_evaluation_id, end_evaluation_id, updated_at
            FROM hypothesis_metrics
            WHERE hypothesis_id IN ({placeholders})
            ORDER BY corr DESC, mmc DESC, hypothesis_id ASC
            """,
            tuple(hypothesis_ids),
        ).fetchall()
        return [_row_to_hypothesis_metric(row) for row in rows if row is not None]

    def _refresh_hypothesis_metrics(
        self,
        *,
        hypothesis_id: str,
        asset: str,
        target: str,
        recorded_at: str,
        window_size: int = DEFAULT_METRIC_WINDOW,
    ) -> None:
        rows = self.conn.execute(
            """
            SELECT p.evaluation_id, p.value AS prediction_value, o.value AS observation_value
            FROM predictions AS p
            JOIN observations AS o ON o.evaluation_id = p.evaluation_id
            JOIN hypotheses AS h ON h.hypothesis_id = p.hypothesis_id
            WHERE p.hypothesis_id = ? AND h.asset = ? AND h.target = ?
            ORDER BY p.evaluation_id DESC
            LIMIT ?
            """,
            (hypothesis_id, asset, target, int(window_size)),
        ).fetchall()
        if not rows:
            self.conn.execute(
                """
                INSERT INTO hypothesis_metrics (
                    hypothesis_id, corr, mmc, sample_count, window_size,
                    start_evaluation_id, end_evaluation_id, updated_at
                )
                VALUES (?, 0.0, 0.0, 0, ?, NULL, NULL, ?)
                ON CONFLICT(hypothesis_id) DO UPDATE SET
                    corr = excluded.corr,
                    mmc = excluded.mmc,
                    sample_count = excluded.sample_count,
                    window_size = excluded.window_size,
                    start_evaluation_id = excluded.start_evaluation_id,
                    end_evaluation_id = excluded.end_evaluation_id,
                    updated_at = excluded.updated_at
                """,
                (hypothesis_id, int(window_size), recorded_at),
            )
            return

        rows = list(reversed(rows))
        evaluation_ids = [str(row["evaluation_id"]) for row in rows]
        predictions = pd.Series(
            [float(row["prediction_value"]) for row in rows],
            index=evaluation_ids,
            dtype=float,
        )
        observations = pd.Series(
            [float(row["observation_value"]) for row in rows],
            index=evaluation_ids,
            dtype=float,
        )

        meta_model = None
        placeholders = ", ".join("?" for _ in evaluation_ids)
        peer_rows = self.conn.execute(
            f"""
            SELECT p.evaluation_id, AVG(p.value) AS meta_prediction
            FROM predictions AS p
            JOIN hypotheses AS h ON h.hypothesis_id = p.hypothesis_id
            WHERE p.evaluation_id IN ({placeholders})
              AND h.asset = ?
              AND h.target = ?
              AND p.hypothesis_id <> ?
            GROUP BY p.evaluation_id
            ORDER BY p.evaluation_id ASC
            """,
            tuple(evaluation_ids) + (asset, target, hypothesis_id),
        ).fetchall()
        if peer_rows:
            meta_model = pd.Series(
                [float(row["meta_prediction"]) for row in peer_rows],
                index=[str(row["evaluation_id"]) for row in peer_rows],
                dtype=float,
            )

        metrics = compute_hypothesis_metrics(
            predictions=predictions,
            target=observations,
            meta_model=meta_model,
            window_size=window_size,
        )
        self.conn.execute(
            """
            INSERT INTO hypothesis_metrics (
                hypothesis_id, corr, mmc, sample_count, window_size,
                start_evaluation_id, end_evaluation_id, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hypothesis_id) DO UPDATE SET
                corr = excluded.corr,
                mmc = excluded.mmc,
                sample_count = excluded.sample_count,
                window_size = excluded.window_size,
                start_evaluation_id = excluded.start_evaluation_id,
                end_evaluation_id = excluded.end_evaluation_id,
                updated_at = excluded.updated_at
            """,
            (
                hypothesis_id,
                metrics.corr,
                metrics.mmc,
                metrics.sample_count,
                metrics.window_size,
                evaluation_ids[0],
                evaluation_ids[-1],
                recorded_at,
            ),
        )

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
            raise ValueError(f"hypothesis is not registered: {hypothesis_id}")
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
        target: str = DEFAULT_TARGET,
    ) -> tuple[HypothesisState, bool]:
        self.ensure_schema()
        existing = self.get_hypothesis(hypothesis_id)
        if existing is not None:
            return existing, False

        timestamp = recorded_at or _utc_now()
        definition = find_hypothesis_definition(hypothesis_id)
        kind = None if definition is None else definition.kind
        signal_name = None if definition is None else definition.signal_name
        lookback = None if definition is None else definition.lookback
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO hypotheses (
                    hypothesis_id, asset, target, kind, signal_name, lookback, status,
                    prediction_count, observation_count, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'registered', 0, 0, ?, ?)
                """,
                (
                    hypothesis_id,
                    asset,
                    target,
                    kind,
                    signal_name,
                    lookback,
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
            SELECT evaluation_id, asset, target, hypothesis_id, prediction_value,
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
            SELECT evaluation_id, hypothesis_id, asset, target, value, recorded_at
            FROM predictions
            WHERE evaluation_id = ? AND hypothesis_id = ?
            """,
            (evaluation_id, hypothesis_id),
        ).fetchone()
        return _row_to_prediction(row)

    def get_observation(self, evaluation_id: str) -> ObservationRecord | None:
        row = self.conn.execute(
            """
            SELECT evaluation_id, asset, target, value, recorded_at
            FROM observations
            WHERE evaluation_id = ?
            """,
            (evaluation_id,),
        ).fetchone()
        return _row_to_observation(row)

    def list_evaluation_snapshots(self, *, limit: int = 20) -> list[EvaluationSnapshot]:
        rows = self.conn.execute(
            """
            SELECT evaluation_id, asset, target, hypothesis_id, prediction_value,
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
        target: str = DEFAULT_TARGET,
    ) -> tuple[PredictionRecord, bool]:
        self.ensure_schema()
        hypothesis = self.get_hypothesis(hypothesis_id)
        if hypothesis is None:
            raise ValueError(
                f"hypothesis must be registered before recording predictions: {hypothesis_id}"
            )
        if hypothesis.status in {"paused", "retired"}:
            raise ValueError(
                f"prediction cannot be recorded while hypothesis is {hypothesis.status}: "
                f"{hypothesis_id}"
            )

        existing = self.get_prediction(evaluation_id, hypothesis_id)
        if existing is not None:
            if existing.value != float(prediction_value):
                raise ValueError(
                    "prediction already exists for this evaluation_id and hypothesis_id with a "
                    "different value"
                )
            return existing, False

        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO predictions (
                    evaluation_id, hypothesis_id, asset, target, value, recorded_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (evaluation_id, hypothesis_id, asset, target, float(prediction_value), timestamp),
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
        target: str = DEFAULT_TARGET,
    ) -> tuple[ObservationRecord, bool]:
        self.ensure_schema()
        existing = self.get_observation(evaluation_id)
        if existing is not None:
            if existing.value != float(observation_value):
                raise ValueError(
                    "observation already exists for this evaluation_id with a different value"
                )
            return existing, False

        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO observations (evaluation_id, asset, target, value, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (evaluation_id, asset, target, float(observation_value), timestamp),
            )

        observation = self.get_observation(evaluation_id)
        assert observation is not None
        return observation, True

    def update_state(
        self,
        *,
        evaluation_id: str,
        hypothesis_id: str,
        recorded_at: str | None = None,
        asset: str = DEFAULT_ASSET,
        target: str = DEFAULT_TARGET,
        input_source: str | None = None,
        input_range_start: str | None = None,
        input_range_end: str | None = None,
        signal_name: str | None = None,
    ) -> tuple[EvaluationSnapshot, bool]:
        self.ensure_schema()
        existing = self.get_evaluation_snapshot(evaluation_id, hypothesis_id)
        if existing is not None:
            return existing, False

        timestamp = recorded_at or _utc_now()
        hypothesis = self.get_hypothesis(hypothesis_id)
        if hypothesis is None:
            raise ValueError(
                f"hypothesis must be registered before updating state: {hypothesis_id}"
            )
        if hypothesis.status in {"paused", "retired"}:
            raise ValueError(
                f"state cannot be updated while hypothesis is {hypothesis.status}: {hypothesis_id}"
            )

        prediction = self.get_prediction(evaluation_id, hypothesis_id)
        if prediction is None:
            raise ValueError(
                f"prediction must be recorded before updating state: {evaluation_id} / {hypothesis_id}"
            )
        observation = self.get_observation(evaluation_id)
        if observation is None:
            raise ValueError(
                f"observation must be finalized before updating state: {evaluation_id}"
            )

        signed_edge = float(prediction.value) * float(observation.value)
        absolute_error = abs(float(prediction.value) - float(observation.value))

        with self.conn:
            self.conn.execute(
                """
                UPDATE hypotheses
                SET prediction_count = prediction_count + 1,
                    observation_count = observation_count + 1,
                    updated_at = ?
                WHERE hypothesis_id = ?
                """,
                (timestamp, hypothesis_id),
            )
            self.conn.execute(
                """
                INSERT INTO evaluation_snapshots (
                    evaluation_id, asset, target, hypothesis_id, prediction_value,
                    observation_value, signed_edge, absolute_error, input_source,
                    input_range_start, input_range_end, signal_name, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    asset,
                    target,
                    hypothesis_id,
                    prediction.value,
                    observation.value,
                    signed_edge,
                    absolute_error,
                    input_source,
                    input_range_start,
                    input_range_end,
                    signal_name,
                    timestamp,
                ),
            )
            self._refresh_hypothesis_metrics(
                hypothesis_id=hypothesis_id,
                asset=asset,
                target=target,
                recorded_at=timestamp,
            )

        snapshot = self.get_evaluation_snapshot(evaluation_id, hypothesis_id)
        assert snapshot is not None
        return snapshot, True

    def run_cycle(
        self,
        *,
        evaluation_id: str,
        hypothesis_id: str,
        prediction_value: float,
        observation_value: float,
        recorded_at: str | None = None,
        asset: str = DEFAULT_ASSET,
        target: str = DEFAULT_TARGET,
        input_source: str | None = None,
        input_range_start: str | None = None,
        input_range_end: str | None = None,
        signal_name: str | None = None,
    ) -> tuple[EvaluationSnapshot, bool]:
        self.ensure_schema()
        self.record_prediction(
            evaluation_id=evaluation_id,
            hypothesis_id=hypothesis_id,
            prediction_value=prediction_value,
            recorded_at=recorded_at,
            asset=asset,
            target=target,
        )
        self.finalize_observation(
            evaluation_id=evaluation_id,
            observation_value=observation_value,
            recorded_at=recorded_at,
            asset=asset,
            target=target,
        )
        return self.update_state(
            evaluation_id=evaluation_id,
            hypothesis_id=hypothesis_id,
            recorded_at=recorded_at,
            asset=asset,
            target=target,
            input_source=input_source,
            input_range_start=input_range_start,
            input_range_end=input_range_end,
            signal_name=signal_name,
        )
