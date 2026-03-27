from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import DEFAULT_ASSET, DEFAULT_TARGET
from .policy import build_cycle_update


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class HypothesisState:
    hypothesis_id: str
    asset: str
    target: str
    status: str
    quality: float
    allocation_trust: float
    prediction_count: int
    observation_count: int


@dataclass(frozen=True)
class CycleSnapshot:
    cycle_id: str
    asset: str
    target: str
    hypothesis_id: str
    prediction_value: float
    observation_value: float
    signed_edge: float
    absolute_error: float
    quality_before: float
    quality_after: float
    quality_delta: float
    allocation_trust_before: float
    allocation_trust_after: float
    allocation_trust_delta: float
    generated_weight: float
    input_source: str | None
    input_range_start: str | None
    input_range_end: str | None
    signal_name: str | None
    created_at: str


@dataclass(frozen=True)
class PredictionRecord:
    cycle_id: str
    hypothesis_id: str
    asset: str
    target: str
    value: float
    recorded_at: str


@dataclass(frozen=True)
class ObservationRecord:
    cycle_id: str
    asset: str
    target: str
    value: float
    recorded_at: str


@dataclass(frozen=True)
class SleeveState:
    asset: str
    target: str
    live_hypothesis_count: int
    mean_quality: float
    total_allocation_trust: float
    latest_cycle_id: str | None
    updated_at: str


def _row_to_hypothesis(row: sqlite3.Row | None) -> HypothesisState | None:
    if row is None:
        return None
    return HypothesisState(
        hypothesis_id=str(row["hypothesis_id"]),
        asset=str(row["asset"]),
        target=str(row["target"]),
        status=str(row["status"]),
        quality=float(row["quality"]),
        allocation_trust=float(row["allocation_trust"]),
        prediction_count=int(row["prediction_count"]),
        observation_count=int(row["observation_count"]),
    )


def _row_to_snapshot(row: sqlite3.Row | None) -> CycleSnapshot | None:
    if row is None:
        return None
    return CycleSnapshot(
        cycle_id=str(row["cycle_id"]),
        asset=str(row["asset"]),
        target=str(row["target"]),
        hypothesis_id=str(row["hypothesis_id"]),
        prediction_value=float(row["prediction_value"]),
        observation_value=float(row["observation_value"]),
        signed_edge=float(row["signed_edge"]),
        absolute_error=float(row["absolute_error"]),
        quality_before=float(row["quality_before"]),
        quality_after=float(row["quality_after"]),
        quality_delta=float(row["quality_delta"]),
        allocation_trust_before=float(row["allocation_trust_before"]),
        allocation_trust_after=float(row["allocation_trust_after"]),
        allocation_trust_delta=float(row["allocation_trust_delta"]),
        generated_weight=float(row["generated_weight"]),
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


def _row_to_sleeve_state(row: sqlite3.Row | None) -> SleeveState | None:
    if row is None:
        return None
    latest_cycle_id = row["latest_cycle_id"]
    return SleeveState(
        asset=str(row["asset"]),
        target=str(row["target"]),
        live_hypothesis_count=int(row["live_hypothesis_count"]),
        mean_quality=float(row["mean_quality"]),
        total_allocation_trust=float(row["total_allocation_trust"]),
        latest_cycle_id=None if latest_cycle_id is None else str(latest_cycle_id),
        updated_at=str(row["updated_at"]),
    )


def _row_to_prediction(row: sqlite3.Row | None) -> PredictionRecord | None:
    if row is None:
        return None
    return PredictionRecord(
        cycle_id=str(row["cycle_id"]),
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
        cycle_id=str(row["cycle_id"]),
        asset=str(row["asset"]),
        target=str(row["target"]),
        value=float(row["value"]),
        recorded_at=str(row["recorded_at"]),
    )


class V1Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def _ensure_cycle_snapshot_column(self, name: str, definition: str) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute("PRAGMA table_info(cycle_snapshots)").fetchall()
        }
        if name not in columns:
            self.conn.execute(f"ALTER TABLE cycle_snapshots ADD COLUMN {name} {definition}")

    def ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                asset TEXT NOT NULL,
                target TEXT NOT NULL,
                status TEXT NOT NULL,
                quality REAL NOT NULL,
                allocation_trust REAL NOT NULL,
                prediction_count INTEGER NOT NULL,
                observation_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                cycle_id TEXT NOT NULL,
                hypothesis_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target TEXT NOT NULL,
                value REAL NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (cycle_id, hypothesis_id)
            );

            CREATE TABLE IF NOT EXISTS observations (
                cycle_id TEXT PRIMARY KEY,
                asset TEXT NOT NULL,
                target TEXT NOT NULL,
                value REAL NOT NULL,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sleeve_state (
                asset TEXT NOT NULL,
                target TEXT NOT NULL,
                live_hypothesis_count INTEGER NOT NULL,
                mean_quality REAL NOT NULL,
                total_allocation_trust REAL NOT NULL,
                latest_cycle_id TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (asset, target)
            );

            CREATE TABLE IF NOT EXISTS cycle_snapshots (
                cycle_id TEXT PRIMARY KEY,
                asset TEXT NOT NULL,
                target TEXT NOT NULL,
                hypothesis_id TEXT NOT NULL,
                prediction_value REAL NOT NULL,
                observation_value REAL NOT NULL,
                signed_edge REAL NOT NULL,
                absolute_error REAL NOT NULL,
                quality_before REAL NOT NULL,
                quality_after REAL NOT NULL,
                quality_delta REAL NOT NULL DEFAULT 0.0,
                allocation_trust_before REAL NOT NULL,
                allocation_trust_after REAL NOT NULL,
                allocation_trust_delta REAL NOT NULL DEFAULT 0.0,
                generated_weight REAL NOT NULL,
                input_source TEXT,
                input_range_start TEXT,
                input_range_end TEXT,
                signal_name TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        self._ensure_cycle_snapshot_column("quality_delta", "REAL NOT NULL DEFAULT 0.0")
        self._ensure_cycle_snapshot_column(
            "allocation_trust_delta",
            "REAL NOT NULL DEFAULT 0.0",
        )
        self._ensure_cycle_snapshot_column("input_source", "TEXT")
        self._ensure_cycle_snapshot_column("input_range_start", "TEXT")
        self._ensure_cycle_snapshot_column("input_range_end", "TEXT")
        self._ensure_cycle_snapshot_column("signal_name", "TEXT")
        self.conn.commit()

    def get_hypothesis(self, hypothesis_id: str) -> HypothesisState | None:
        row = self.conn.execute(
            """
            SELECT hypothesis_id, asset, target, quality, allocation_trust,
                   status, prediction_count, observation_count
            FROM hypotheses
            WHERE hypothesis_id = ?
            """,
            (hypothesis_id,),
        ).fetchone()
        return _row_to_hypothesis(row)

    def set_hypothesis_status(
        self,
        hypothesis_id: str,
        *,
        new_status: str,
        allowed_from: tuple[str, ...],
        recorded_at: str | None = None,
    ) -> HypothesisState:
        self.ensure_schema()
        hypothesis = self.get_hypothesis(hypothesis_id)
        if hypothesis is None:
            raise ValueError(f"hypothesis is not registered: {hypothesis_id}")
        if hypothesis.status not in allowed_from:
            allowed = ", ".join(allowed_from)
            raise ValueError(
                f"invalid hypothesis transition for {hypothesis_id}: "
                f"{hypothesis.status} -> {new_status} (allowed from: {allowed})"
            )

        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                UPDATE hypotheses
                SET status = ?, updated_at = ?
                WHERE hypothesis_id = ?
                """,
                (new_status, timestamp, hypothesis_id),
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
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO hypotheses (
                    hypothesis_id, asset, target, status, quality,
                    allocation_trust, prediction_count, observation_count,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, 'registered', 0.0, 0.0, 0, 0, ?, ?)
                """,
                (hypothesis_id, asset, target, timestamp, timestamp),
            )

        hypothesis = self.get_hypothesis(hypothesis_id)
        assert hypothesis is not None
        return hypothesis, True

    def get_cycle_snapshot(self, cycle_id: str) -> CycleSnapshot | None:
        row = self.conn.execute(
            """
            SELECT cycle_id, asset, target, hypothesis_id, prediction_value,
                   observation_value, signed_edge, absolute_error, quality_before,
                   quality_after, quality_delta, allocation_trust_before,
                   allocation_trust_after, allocation_trust_delta, generated_weight,
                   input_source, input_range_start, input_range_end, signal_name,
                   created_at
            FROM cycle_snapshots
            WHERE cycle_id = ?
            """,
            (cycle_id,),
        ).fetchone()
        return _row_to_snapshot(row)

    def get_prediction(self, cycle_id: str, hypothesis_id: str) -> PredictionRecord | None:
        row = self.conn.execute(
            """
            SELECT cycle_id, hypothesis_id, asset, target, value, recorded_at
            FROM predictions
            WHERE cycle_id = ? AND hypothesis_id = ?
            """,
            (cycle_id, hypothesis_id),
        ).fetchone()
        return _row_to_prediction(row)

    def get_observation(self, cycle_id: str) -> ObservationRecord | None:
        row = self.conn.execute(
            """
            SELECT cycle_id, asset, target, value, recorded_at
            FROM observations
            WHERE cycle_id = ?
            """,
            (cycle_id,),
        ).fetchone()
        return _row_to_observation(row)

    def list_cycle_snapshots(self, *, limit: int = 20) -> list[CycleSnapshot]:
        rows = self.conn.execute(
            """
            SELECT cycle_id, asset, target, hypothesis_id, prediction_value,
                   observation_value, signed_edge, absolute_error, quality_before,
                   quality_after, quality_delta, allocation_trust_before,
                   allocation_trust_after, allocation_trust_delta, generated_weight,
                   input_source, input_range_start, input_range_end, signal_name,
                   created_at
            FROM cycle_snapshots
            ORDER BY created_at DESC, cycle_id DESC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [_row_to_snapshot(row) for row in rows]

    def get_sleeve_state(
        self,
        *,
        asset: str = DEFAULT_ASSET,
        target: str = DEFAULT_TARGET,
    ) -> SleeveState | None:
        row = self.conn.execute(
            """
            SELECT asset, target, live_hypothesis_count, mean_quality,
                   total_allocation_trust, latest_cycle_id, updated_at
            FROM sleeve_state
            WHERE asset = ? AND target = ?
            """,
            (asset, target),
        ).fetchone()
        return _row_to_sleeve_state(row)

    def record_prediction(
        self,
        *,
        cycle_id: str,
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

        existing = self.get_prediction(cycle_id, hypothesis_id)
        if existing is not None:
            if existing.value != float(prediction_value):
                raise ValueError(
                    "prediction already exists for this cycle_id and hypothesis_id with a "
                    "different value"
                )
            return existing, False

        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO predictions (
                    cycle_id, hypothesis_id, asset, target, value, recorded_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cycle_id, hypothesis_id, asset, target, float(prediction_value), timestamp),
            )

        prediction = self.get_prediction(cycle_id, hypothesis_id)
        assert prediction is not None
        return prediction, True

    def finalize_observation(
        self,
        *,
        cycle_id: str,
        observation_value: float,
        recorded_at: str | None = None,
        asset: str = DEFAULT_ASSET,
        target: str = DEFAULT_TARGET,
    ) -> tuple[ObservationRecord, bool]:
        self.ensure_schema()
        existing = self.get_observation(cycle_id)
        if existing is not None:
            if existing.value != float(observation_value):
                raise ValueError(
                    "observation already exists for this cycle_id with a different value"
                )
            return existing, False

        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO observations (cycle_id, asset, target, value, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cycle_id, asset, target, float(observation_value), timestamp),
            )

        observation = self.get_observation(cycle_id)
        assert observation is not None
        return observation, True

    def update_state(
        self,
        *,
        cycle_id: str,
        hypothesis_id: str,
        recorded_at: str | None = None,
        asset: str = DEFAULT_ASSET,
        target: str = DEFAULT_TARGET,
        input_source: str | None = None,
        input_range_start: str | None = None,
        input_range_end: str | None = None,
        signal_name: str | None = None,
    ) -> tuple[CycleSnapshot, bool]:
        self.ensure_schema()
        existing = self.get_cycle_snapshot(cycle_id)
        if existing is not None:
            return existing, False

        timestamp = recorded_at or _utc_now()
        before_state = self.get_hypothesis(hypothesis_id)
        if before_state is None:
            raise ValueError(
                f"hypothesis must be registered before updating state: {hypothesis_id}"
            )
        if before_state.status in {"paused", "retired"}:
            raise ValueError(
                f"state cannot be updated while hypothesis is {before_state.status}: "
                f"{hypothesis_id}"
            )
        prediction = self.get_prediction(cycle_id, hypothesis_id)
        if prediction is None:
            raise ValueError(
                f"prediction must be recorded before updating state: {cycle_id} / {hypothesis_id}"
            )
        observation = self.get_observation(cycle_id)
        if observation is None:
            raise ValueError(f"observation must be finalized before updating state: {cycle_id}")

        quality_before = before_state.quality
        trust_before = before_state.allocation_trust
        update = build_cycle_update(
            quality_before=quality_before,
            allocation_trust_before=trust_before,
            prediction_value=prediction.value,
            observation_value=observation.value,
        )

        next_status = "live" if update.allocation_trust_after > 0.0 else "active"

        with self.conn:
            self.conn.execute(
                """
                UPDATE hypotheses
                SET status = ?,
                    quality = ?,
                    allocation_trust = ?,
                    prediction_count = prediction_count + 1,
                    observation_count = observation_count + 1,
                    updated_at = ?
                WHERE hypothesis_id = ?
                """,
                (
                    next_status,
                    update.quality_after,
                    update.allocation_trust_after,
                    timestamp,
                    hypothesis_id,
                ),
            )

            totals_row = self.conn.execute(
                """
                SELECT
                    COALESCE(SUM(allocation_trust), 0.0) AS total_trust,
                    COALESCE(AVG(quality), 0.0) AS mean_quality,
                    SUM(CASE WHEN status = 'live' THEN 1 ELSE 0 END) AS live_count
                FROM hypotheses
                WHERE asset = ? AND target = ?
                """,
                (asset, target),
            ).fetchone()
            assert totals_row is not None
            total_trust = float(totals_row["total_trust"])
            mean_quality = float(totals_row["mean_quality"])
            live_count = int(totals_row["live_count"] or 0)
            generated_weight = (
                0.0 if total_trust <= 0.0 else update.allocation_trust_after / total_trust
            )

            self.conn.execute(
                """
                INSERT INTO cycle_snapshots (
                    cycle_id, asset, target, hypothesis_id, prediction_value,
                    observation_value, signed_edge, absolute_error, quality_before,
                    quality_after, quality_delta, allocation_trust_before,
                    allocation_trust_after, allocation_trust_delta, generated_weight,
                    input_source, input_range_start, input_range_end, signal_name,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle_id,
                    asset,
                    target,
                    hypothesis_id,
                    prediction.value,
                    observation.value,
                    update.signed_edge,
                    update.absolute_error,
                    update.quality_before,
                    update.quality_after,
                    update.quality_delta,
                    update.allocation_trust_before,
                    update.allocation_trust_after,
                    update.allocation_trust_delta,
                    generated_weight,
                    input_source,
                    input_range_start,
                    input_range_end,
                    signal_name,
                    timestamp,
                ),
            )
            self.conn.execute(
                """
                INSERT INTO sleeve_state (
                    asset, target, live_hypothesis_count, mean_quality,
                    total_allocation_trust, latest_cycle_id, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset, target) DO UPDATE SET
                    live_hypothesis_count = excluded.live_hypothesis_count,
                    mean_quality = excluded.mean_quality,
                    total_allocation_trust = excluded.total_allocation_trust,
                    latest_cycle_id = excluded.latest_cycle_id,
                    updated_at = excluded.updated_at
                """,
                (asset, target, live_count, mean_quality, total_trust, cycle_id, timestamp),
            )

        snapshot = self.get_cycle_snapshot(cycle_id)
        assert snapshot is not None
        return snapshot, True

    def run_cycle(
        self,
        *,
        cycle_id: str,
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
    ) -> tuple[CycleSnapshot, bool]:
        self.ensure_schema()
        self.record_prediction(
            cycle_id=cycle_id,
            hypothesis_id=hypothesis_id,
            prediction_value=prediction_value,
            recorded_at=recorded_at,
            asset=asset,
            target=target,
        )
        self.finalize_observation(
            cycle_id=cycle_id,
            observation_value=observation_value,
            recorded_at=recorded_at,
            asset=asset,
            target=target,
        )
        return self.update_state(
            cycle_id=cycle_id,
            hypothesis_id=hypothesis_id,
            recorded_at=recorded_at,
            asset=asset,
            target=target,
            input_source=input_source,
            input_range_start=input_range_start,
            input_range_end=input_range_end,
            signal_name=signal_name,
        )
