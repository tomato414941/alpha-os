from __future__ import annotations

from datetime import UTC, datetime

from .config import DEFAULT_ASSET, DEFAULT_TARGET
from .metrics_service import refresh_hypothesis_metrics
from .store import EvaluationSnapshot, EvaluationStore


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def update_evaluation_state(
    store: EvaluationStore,
    *,
    evaluation_id: str,
    hypothesis_id: str,
    recorded_at: str | None = None,
    asset: str = DEFAULT_ASSET,
    target_id: str = DEFAULT_TARGET,
    input_source: str | None = None,
    input_range_start: str | None = None,
    input_range_end: str | None = None,
    signal_name: str | None = None,
) -> tuple[EvaluationSnapshot, bool]:
    store.ensure_schema()
    existing = store.get_evaluation_snapshot(evaluation_id, hypothesis_id)
    if existing is not None:
        return existing, False

    timestamp = recorded_at or _utc_now()
    hypothesis = store.get_hypothesis(hypothesis_id)
    if hypothesis is None:
        raise ValueError(f"hypothesis must exist before updating state: {hypothesis_id}")
    if hypothesis.status != "active":
        raise ValueError(
            f"state cannot be updated while hypothesis is {hypothesis.status}: {hypothesis_id}"
        )
    if hypothesis.asset != asset:
        raise ValueError(
            f"evaluation asset does not match hypothesis asset: {asset} != {hypothesis.asset}"
        )
    if hypothesis.target_id != target_id:
        raise ValueError(
            "evaluation target does not match hypothesis target: "
            f"{target_id} != {hypothesis.target_id}"
        )

    prediction = store.get_prediction(evaluation_id, hypothesis_id)
    if prediction is None:
        raise ValueError(
            f"prediction must be recorded before updating state: {evaluation_id} / {hypothesis_id}"
        )
    if prediction.asset != asset or prediction.target_id != target_id:
        raise ValueError(
            "recorded prediction asset/target does not match evaluation update request"
        )
    observation = store.get_observation(evaluation_id)
    if observation is None:
        raise ValueError(
            f"observation must be finalized before updating state: {evaluation_id}"
        )
    if observation.asset != asset or observation.target_id != target_id:
        raise ValueError(
            "recorded observation asset/target does not match evaluation update request"
        )

    signed_edge = float(prediction.value) * float(observation.value)
    absolute_error = abs(float(prediction.value) - float(observation.value))

    with store.conn:
        store.conn.execute(
            """
            UPDATE hypotheses
            SET prediction_count = prediction_count + 1,
                observation_count = observation_count + 1,
                updated_at = ?
            WHERE hypothesis_id = ?
            """,
            (timestamp, hypothesis_id),
        )
        store.conn.execute(
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
                target_id,
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
        refresh_hypothesis_metrics(
            store,
            hypothesis_id=hypothesis_id,
            asset=asset,
            target_id=target_id,
            recorded_at=timestamp,
        )

    snapshot = store.get_evaluation_snapshot(evaluation_id, hypothesis_id)
    assert snapshot is not None
    return snapshot, True


def apply_evaluation(
    store: EvaluationStore,
    *,
    evaluation_id: str,
    hypothesis_id: str,
    prediction_value: float,
    observation_value: float,
    recorded_at: str | None = None,
    asset: str = DEFAULT_ASSET,
    target_id: str = DEFAULT_TARGET,
    input_source: str | None = None,
    input_range_start: str | None = None,
    input_range_end: str | None = None,
    signal_name: str | None = None,
) -> tuple[EvaluationSnapshot, bool]:
    store.ensure_schema()
    store.record_prediction(
        evaluation_id=evaluation_id,
        hypothesis_id=hypothesis_id,
        prediction_value=prediction_value,
        recorded_at=recorded_at,
        asset=asset,
        target_id=target_id,
    )
    store.finalize_observation(
        evaluation_id=evaluation_id,
        observation_value=observation_value,
        recorded_at=recorded_at,
        asset=asset,
        target_id=target_id,
    )
    return update_evaluation_state(
        store,
        evaluation_id=evaluation_id,
        hypothesis_id=hypothesis_id,
        recorded_at=recorded_at,
        asset=asset,
        target_id=target_id,
        input_source=input_source,
        input_range_start=input_range_start,
        input_range_end=input_range_end,
        signal_name=signal_name,
    )
