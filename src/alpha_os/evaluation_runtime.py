from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from .config import DEFAULT_TARGET
from .evaluation_inputs import EvaluationInput
from .signal_registry import executable_signal_from_document
from .portfolio_decision import ObservationSpec
from .metrics_service import refresh_target_metrics
from .meta_aggregation_service import refresh_target_meta_predictions
from .meta_metrics_service import refresh_target_meta_prediction_metrics
from .store import EvaluationSnapshot, EvaluationStore


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _default_evaluation_id(*, subject_id: str, target_id: str, date: str) -> str:
    return f"{subject_id}:{target_id}:{date}"


@dataclass(frozen=True)
class _PreparedBatchEvaluation:
    evaluation_id: str
    signal_id: str
    subject_id: str
    asset: str
    target_id: str
    prediction_value: float
    observation_value: float
    funding_cost_bps: float | None
    borrow_fee_bps: float | None
    roll_cost_bps: float | None
    financing_cost_bps: float | None
    contract_multiplier: float | None
    contract_id: str | None
    contract_family: str | None
    quote_ccy: str | None
    collateral_ccy: str | None
    roll_event_json: str | None
    observation_spec_id: str | None
    observable_id: str | None
    adapter_kind: str | None
    recorded_at: str

def update_evaluation_state(
    store: EvaluationStore,
    *,
    evaluation_id: str,

    signal_id: str | None = None,
    recorded_at: str | None = None,
    target_id: str = DEFAULT_TARGET,
    subject_id: str | None = None,
    input_source: str | None = None,
    input_range_start: str | None = None,
    input_range_end: str | None = None,
    funding_cost_bps: float | None = None,
    borrow_fee_bps: float | None = None,
    roll_cost_bps: float | None = None,
    financing_cost_bps: float | None = None,
    contract_multiplier: float | None = None,
    contract_id: str | None = None,
    contract_family: str | None = None,
    quote_ccy: str | None = None,
    collateral_ccy: str | None = None,
    roll_event: dict[str, object] | None = None,
    observation_spec: ObservationSpec | None = None,
    refresh_metrics: bool = True,
) -> tuple[EvaluationSnapshot, bool]:
    store.ensure_schema()
    resolved_signal_id = signal_id
    if resolved_signal_id is None:
        raise ValueError("evaluation update requires signal_id")
    existing = store.get_evaluation_snapshot(evaluation_id, resolved_signal_id)
    if existing is not None:
        return existing, False

    timestamp = recorded_at or _utc_now()
    signal = store.get_signal(resolved_signal_id)
    if signal is None:
        raise ValueError(
            "signal must exist before updating state: "
            f"{resolved_signal_id}"
        )
    executable = executable_signal_from_document(
        signal_id=signal.signal_id,
        asset=signal.asset,
        document=signal.definition,
        target_id=signal.target_id,
    )
    effective_observation_spec = (
        executable.definition.observation_spec
        if observation_spec is None
        else observation_spec
    )
    if signal.status != "active":
        raise ValueError(
            "state cannot be updated while signal is "
            f"{signal.status}: {resolved_signal_id}"
        )
    if signal.target_id != target_id:
        raise ValueError(
            "evaluation target does not match signal target: "
            f"{target_id} != {signal.target_id}"
        )
    if subject_id is not None and executable.subject_id != subject_id:
        raise ValueError(
            "evaluation subject does not match signal subject: "
            f"{subject_id} != {executable.subject_id}"
        )

    prediction = store.get_prediction(evaluation_id, resolved_signal_id)
    if prediction is None:
        raise ValueError(
            "prediction must be recorded before updating state: "
            f"{evaluation_id} / {resolved_signal_id}"
        )
    if (
        prediction.subject_id != executable.subject_id
        or prediction.asset != executable.asset
        or prediction.target_id != target_id
    ):
        raise ValueError(
            "recorded prediction asset/target does not match evaluation update request"
        )
    observation = store.get_observation(evaluation_id)
    if observation is None:
        raise ValueError(
            f"observation must be finalized before updating state: {evaluation_id}"
        )
    if (
        observation.subject_id != executable.subject_id
        or observation.asset != executable.asset
        or observation.target_id != target_id
    ):
        raise ValueError(
            "recorded observation asset/target does not match evaluation update request"
        )

    signed_edge = float(prediction.value) * float(observation.value)
    absolute_error = abs(float(prediction.value) - float(observation.value))

    with store.conn:
        store.conn.execute(
            """
            UPDATE signals
            SET prediction_count = prediction_count + 1,
                observation_count = observation_count + 1,
                updated_at = ?
            WHERE signal_id = ?
            """,
            (timestamp, resolved_signal_id),
        )
        store.conn.execute(
            """
            INSERT INTO evaluation_snapshots (
                evaluation_id, subject_id, asset, target_id, signal_id, prediction_value,
                observation_value, signed_edge, absolute_error, input_source,
                input_range_start, input_range_end,
                funding_cost_bps, borrow_fee_bps, roll_cost_bps, financing_cost_bps,
                contract_multiplier, contract_id, contract_family, quote_ccy,
                collateral_ccy, roll_event_json,
                observation_spec_id, observable_id, adapter_kind,
                signal_name, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation_id,
                executable.subject_id,
                executable.asset,
                target_id,
                resolved_signal_id,
                prediction.value,
                observation.value,
                signed_edge,
                absolute_error,
                input_source,
                input_range_start,
                input_range_end,
                funding_cost_bps,
                borrow_fee_bps,
                roll_cost_bps,
                financing_cost_bps,
                contract_multiplier,
                contract_id,
                contract_family,
                quote_ccy,
                collateral_ccy,
                None if roll_event is None else json.dumps(roll_event, ensure_ascii=True),
                None
                if effective_observation_spec is None
                else effective_observation_spec.observation_spec_id,
                None
                if effective_observation_spec is None
                else effective_observation_spec.observable_id,
                None
                if effective_observation_spec is None
                else effective_observation_spec.adapter_kind,
                None,
                timestamp,
            ),
        )
        if refresh_metrics:
            refresh_target_metrics(
                store,
                subject_id=executable.subject_id,
                asset=executable.asset,
                target_id=target_id,
                recorded_at=timestamp,
            )
            refresh_target_meta_predictions(
                store,
                subject_id=executable.subject_id,
                asset=executable.asset,
                target_id=target_id,
                recorded_at=timestamp,
            )
            refresh_target_meta_prediction_metrics(
                store,
                subject_id=executable.subject_id,
                asset=executable.asset,
                target_id=target_id,
                recorded_at=timestamp,
            )

    snapshot = store.get_evaluation_snapshot(evaluation_id, resolved_signal_id)
    assert snapshot is not None
    return snapshot, True


def apply_evaluation(
    store: EvaluationStore,
    *,
    evaluation_id: str,

    signal_id: str | None = None,
    prediction_value: float,
    observation_value: float,
    recorded_at: str | None = None,
    target_id: str = DEFAULT_TARGET,
    subject_id: str | None = None,
    input_source: str | None = None,
    input_range_start: str | None = None,
    input_range_end: str | None = None,
    funding_cost_bps: float | None = None,
    borrow_fee_bps: float | None = None,
    roll_cost_bps: float | None = None,
    financing_cost_bps: float | None = None,
    contract_multiplier: float | None = None,
    contract_id: str | None = None,
    contract_family: str | None = None,
    quote_ccy: str | None = None,
    collateral_ccy: str | None = None,
    roll_event: dict[str, object] | None = None,
    observation_spec: ObservationSpec | None = None,
    refresh_metrics: bool = True,
) -> tuple[EvaluationSnapshot, bool]:
    store.ensure_schema()
    resolved_signal_id = signal_id
    if resolved_signal_id is None:
        raise ValueError("evaluation apply requires signal_id")
    signal = store.get_signal(resolved_signal_id)
    if signal is None:
        raise ValueError(
            "signal must exist before applying evaluation: "
            f"{resolved_signal_id}"
        )
    executable = executable_signal_from_document(
        signal_id=signal.signal_id,
        asset=signal.asset,
        document=signal.definition,
        target_id=signal.target_id,
    )
    if subject_id is not None and executable.subject_id != subject_id:
        raise ValueError(
            "evaluation subject does not match signal subject: "
            f"{subject_id} != {executable.subject_id}"
        )
    store.record_prediction(
        evaluation_id=evaluation_id,
        signal_id=resolved_signal_id,
        prediction_value=prediction_value,
        recorded_at=recorded_at,
        subject_id=executable.subject_id,
        asset=executable.asset,
        target_id=target_id,
    )
    store.finalize_observation(
        evaluation_id=evaluation_id,
        observation_value=observation_value,
        recorded_at=recorded_at,
        subject_id=executable.subject_id,
        asset=executable.asset,
        target_id=target_id,
    )
    return update_evaluation_state(
        store,
        evaluation_id=evaluation_id,
        signal_id=resolved_signal_id,
        recorded_at=recorded_at,
        target_id=target_id,
        subject_id=executable.subject_id,
        input_source=input_source,
        input_range_start=input_range_start,
        input_range_end=input_range_end,
        funding_cost_bps=funding_cost_bps,
        borrow_fee_bps=borrow_fee_bps,
        roll_cost_bps=roll_cost_bps,
        financing_cost_bps=financing_cost_bps,
        contract_multiplier=contract_multiplier,
        contract_id=contract_id,
        contract_family=contract_family,
        quote_ccy=quote_ccy,
        collateral_ccy=collateral_ccy,
        roll_event=roll_event,
        observation_spec=observation_spec,
        refresh_metrics=refresh_metrics,
    )


def apply_evaluations_batch(
    store: EvaluationStore,
    *,
    evaluation_inputs: list[EvaluationInput],
    recorded_at: str | None = None,
    input_source: str | None = None,
    input_range_start: str | None = None,
    input_range_end: str | None = None,
    refresh_metrics: bool = True,
    refresh_meta_predictions: bool = True,
) -> tuple[EvaluationSnapshot | None, int, int]:
    store.ensure_schema()
    if not evaluation_inputs:
        return None, 0, 0

    timestamp = recorded_at or _utc_now()
    prepared_rows: list[_PreparedBatchEvaluation] = []
    touched_targets: set[tuple[str, str, str]] = set()
    seen_keys: set[tuple[str, str]] = set()
    observation_by_evaluation: dict[str, tuple[str, str, str, float]] = {}
    executable_cache: dict[str, object] = {}

    for evaluation_input in evaluation_inputs:
        signal = store.get_signal(
            evaluation_input.signal_id
        )
        if signal is None:
            raise ValueError(
                "signal must exist before applying evaluation batch: "
                f"{evaluation_input.signal_id}"
            )
        if signal.status != "active":
            raise ValueError(
                "evaluation batch requires active signals: "
                f"{evaluation_input.signal_id} is {signal.status}"
            )
        executable = executable_cache.get(evaluation_input.signal_id)
        if executable is None:
            executable = executable_signal_from_document(
                signal_id=signal.signal_id,
                asset=signal.asset,
                document=signal.definition,
                target_id=signal.target_id,
            )
            executable_cache[evaluation_input.signal_id] = executable
        if signal.target_id != evaluation_input.target_id:
            raise ValueError(
                "evaluation target does not match signal target: "
                f"{evaluation_input.target_id} != {signal.target_id}"
            )
        if executable.subject_id != evaluation_input.subject_id:
            raise ValueError(
                "evaluation subject does not match signal subject: "
                f"{evaluation_input.subject_id} != {executable.subject_id}"
            )
        evaluation_id = evaluation_input.evaluation_id or _default_evaluation_id(
            subject_id=evaluation_input.subject_id,
            target_id=evaluation_input.target_id,
            date=evaluation_input.date,
        )
        evaluation_key = (evaluation_id, evaluation_input.signal_id)
        if evaluation_key in seen_keys:
            raise ValueError(
                "evaluation batch contains duplicate evaluation/signal-candidate pairs: "
                f"{evaluation_id} / {evaluation_input.signal_id}"
            )
        seen_keys.add(evaluation_key)

        observation_signature = (
            executable.subject_id,
            executable.asset,
            evaluation_input.target_id,
            float(evaluation_input.observation),
        )
        existing_signature = observation_by_evaluation.get(evaluation_id)
        if existing_signature is None:
            observation_by_evaluation[evaluation_id] = observation_signature
        elif existing_signature != observation_signature:
            raise ValueError(
                "evaluation batch contains inconsistent observations for one evaluation_id: "
                f"{evaluation_id}"
            )

        observation_spec = executable.definition.observation_spec
        prepared_rows.append(
            _PreparedBatchEvaluation(
                evaluation_id=evaluation_id,
                signal_id=evaluation_input.signal_id,
                subject_id=executable.subject_id,
                asset=executable.asset,
                target_id=evaluation_input.target_id,
                prediction_value=float(evaluation_input.prediction),
                observation_value=float(evaluation_input.observation),
                funding_cost_bps=(
                    None
                    if evaluation_input.funding_cost_bps is None
                    else float(evaluation_input.funding_cost_bps)
                ),
                borrow_fee_bps=(
                    None
                    if evaluation_input.borrow_fee_bps is None
                    else float(evaluation_input.borrow_fee_bps)
                ),
                roll_cost_bps=(
                    None
                    if evaluation_input.roll_cost_bps is None
                    else float(evaluation_input.roll_cost_bps)
                ),
                financing_cost_bps=(
                    None
                    if evaluation_input.financing_cost_bps is None
                    else float(evaluation_input.financing_cost_bps)
                ),
                contract_multiplier=(
                    None
                    if evaluation_input.contract_multiplier is None
                    else float(evaluation_input.contract_multiplier)
                ),
                contract_id=(
                    None
                    if evaluation_input.contract_id is None
                    else str(evaluation_input.contract_id)
                ),
                contract_family=(
                    None
                    if evaluation_input.contract_family is None
                    else str(evaluation_input.contract_family)
                ),
                quote_ccy=(
                    None
                    if evaluation_input.quote_ccy is None
                    else str(evaluation_input.quote_ccy)
                ),
                collateral_ccy=(
                    None
                    if evaluation_input.collateral_ccy is None
                    else str(evaluation_input.collateral_ccy)
                ),
                roll_event_json=(
                    None
                    if evaluation_input.roll_event is None
                    else json.dumps(evaluation_input.roll_event, ensure_ascii=True)
                ),
                observation_spec_id=None
                if observation_spec is None
                else observation_spec.observation_spec_id,
                observable_id=None
                if observation_spec is None
                else observation_spec.observable_id,
                adapter_kind=None
                if observation_spec is None
                else observation_spec.adapter_kind,
                recorded_at=timestamp,
            )
        )
        touched_targets.add(
            (
                executable.subject_id,
                executable.asset,
                evaluation_input.target_id,
            )
        )
        store.register_target(evaluation_input.target_id, recorded_at=timestamp)

    with store.conn:
        store.conn.execute("DROP TABLE IF EXISTS batch_evaluation_inputs")
        store.conn.execute("DROP TABLE IF EXISTS batch_new_snapshots")
        store.conn.execute(
            """
            CREATE TEMP TABLE batch_evaluation_inputs (
                evaluation_id TEXT NOT NULL,
                signal_id TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                prediction_value REAL NOT NULL,
                observation_value REAL NOT NULL,
                funding_cost_bps REAL,
                borrow_fee_bps REAL,
                roll_cost_bps REAL,
                financing_cost_bps REAL,
                contract_multiplier REAL,
                contract_id TEXT,
                contract_family TEXT,
                quote_ccy TEXT,
                collateral_ccy TEXT,
                roll_event_json TEXT,
                observation_spec_id TEXT,
                observable_id TEXT,
                adapter_kind TEXT,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (evaluation_id, signal_id)
            )
            """
        )
        store.conn.executemany(
            """
            INSERT INTO batch_evaluation_inputs (
                evaluation_id, signal_id, subject_id, asset, target_id,
                prediction_value, observation_value,
                funding_cost_bps, borrow_fee_bps, roll_cost_bps, financing_cost_bps,
                contract_multiplier, contract_id, contract_family, quote_ccy,
                collateral_ccy, roll_event_json,
                observation_spec_id, observable_id, adapter_kind, recorded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.evaluation_id,
                    row.signal_id,
                    row.subject_id,
                    row.asset,
                    row.target_id,
                    row.prediction_value,
                    row.observation_value,
                    row.funding_cost_bps,
                    row.borrow_fee_bps,
                    row.roll_cost_bps,
                    row.financing_cost_bps,
                    row.contract_multiplier,
                    row.contract_id,
                    row.contract_family,
                    row.quote_ccy,
                    row.collateral_ccy,
                    row.roll_event_json,
                    row.observation_spec_id,
                    row.observable_id,
                    row.adapter_kind,
                    row.recorded_at,
                )
                for row in prepared_rows
            ],
        )

        prediction_conflict = store.conn.execute(
            """
            SELECT b.evaluation_id, b.signal_id
            FROM batch_evaluation_inputs AS b
            JOIN predictions AS p
              ON p.evaluation_id = b.evaluation_id
             AND p.signal_id = b.signal_id
            WHERE p.subject_id != b.subject_id
               OR p.asset != b.asset
               OR p.target_id != b.target_id
               OR p.value != b.prediction_value
            LIMIT 1
            """
        ).fetchone()
        if prediction_conflict is not None:
            raise ValueError(
                "prediction already exists with different values: "
                f"{prediction_conflict['evaluation_id']} / {prediction_conflict['signal_id']}"
            )

        observation_conflict = store.conn.execute(
            """
            WITH batch_observations AS (
                SELECT evaluation_id, subject_id, asset, target_id, observation_value, recorded_at
                FROM batch_evaluation_inputs
                GROUP BY evaluation_id, subject_id, asset, target_id, observation_value, recorded_at
            )
            SELECT b.evaluation_id
            FROM batch_observations AS b
            JOIN observations AS o
              ON o.evaluation_id = b.evaluation_id
            WHERE o.subject_id != b.subject_id
               OR o.asset != b.asset
               OR o.target_id != b.target_id
               OR o.value != b.observation_value
            LIMIT 1
            """
        ).fetchone()
        if observation_conflict is not None:
            raise ValueError(
                "observation already exists with different values: "
                f"{observation_conflict['evaluation_id']}"
            )

        store.conn.execute(
            """
            CREATE TEMP TABLE batch_new_snapshots AS
            SELECT b.*
            FROM batch_evaluation_inputs AS b
            LEFT JOIN evaluation_snapshots AS s
              ON s.evaluation_id = b.evaluation_id
             AND s.signal_id = b.signal_id
            WHERE s.evaluation_id IS NULL
            """
        )

        store.conn.execute(
            """
            INSERT OR IGNORE INTO predictions (
                evaluation_id, signal_id, subject_id, asset, target_id, value, recorded_at
            )
            SELECT evaluation_id, signal_id, subject_id, asset, target_id,
                   prediction_value, recorded_at
            FROM batch_new_snapshots
            """
        )
        store.conn.execute(
            """
            INSERT OR IGNORE INTO observations (
                evaluation_id, subject_id, asset, target_id, value, recorded_at
            )
            SELECT evaluation_id, subject_id, asset, target_id, observation_value, recorded_at
            FROM (
                SELECT evaluation_id, subject_id, asset, target_id, observation_value, recorded_at
                FROM batch_new_snapshots
                GROUP BY evaluation_id, subject_id, asset, target_id, observation_value, recorded_at
            )
            """
        )
        store.conn.execute(
            """
            INSERT INTO evaluation_snapshots (
                evaluation_id, subject_id, asset, target_id, signal_id, prediction_value,
                observation_value, signed_edge, absolute_error, input_source,
                input_range_start, input_range_end,
                funding_cost_bps, borrow_fee_bps, roll_cost_bps, financing_cost_bps,
                contract_multiplier, contract_id, contract_family, quote_ccy,
                collateral_ccy, roll_event_json,
                observation_spec_id, observable_id, adapter_kind,
                signal_name, created_at
            )
            SELECT evaluation_id, subject_id, asset, target_id, signal_id,
                   prediction_value, observation_value,
                   prediction_value * observation_value,
                   ABS(prediction_value - observation_value),
                   ?, ?, ?,
                   funding_cost_bps, borrow_fee_bps, roll_cost_bps, financing_cost_bps,
                   contract_multiplier, contract_id, contract_family, quote_ccy,
                   collateral_ccy, roll_event_json,
                   observation_spec_id, observable_id, adapter_kind,
                   NULL, recorded_at
            FROM batch_new_snapshots
            """,
            (
                input_source,
                input_range_start,
                input_range_end,
            ),
        )

        signal_counts = store.conn.execute(
            """
            SELECT signal_id, COUNT(*) AS created_count
            FROM batch_new_snapshots
            GROUP BY signal_id
            """
        ).fetchall()
        store.conn.executemany(
            """
            UPDATE signals
            SET prediction_count = prediction_count + ?,
                observation_count = observation_count + ?,
                updated_at = ?
            WHERE signal_id = ?
            """,
            [
                (
                    int(row["created_count"]),
                    int(row["created_count"]),
                    timestamp,
                    str(row["signal_id"]),
                )
                for row in signal_counts
            ],
        )

        created_count = int(
            store.conn.execute(
                "SELECT COUNT(*) AS count FROM batch_new_snapshots"
            ).fetchone()["count"]
        )
        store.conn.execute("DROP TABLE IF EXISTS batch_new_snapshots")
        store.conn.execute("DROP TABLE IF EXISTS batch_evaluation_inputs")

    if refresh_metrics:
        for subject_id, asset, target_id in sorted(touched_targets):
            refresh_target_metrics(
                store,
                subject_id=subject_id,
                asset=asset,
                target_id=target_id,
            )
            if refresh_meta_predictions:
                refresh_target_meta_predictions(
                    store,
                    subject_id=subject_id,
                    asset=asset,
                    target_id=target_id,
                )
                refresh_target_meta_prediction_metrics(
                    store,
                    subject_id=subject_id,
                    asset=asset,
                    target_id=target_id,
                )

    last_row = prepared_rows[-1]
    latest_snapshot = store.get_evaluation_snapshot(
        last_row.evaluation_id,
        last_row.signal_id,
    )
    existing_count = len(prepared_rows) - created_count
    return latest_snapshot, created_count, existing_count
