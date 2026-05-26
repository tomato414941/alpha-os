from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from .config import DEFAULT_SUBJECT_ID, DEFAULT_TARGET, default_runtime_asset
from .scoring import DEFAULT_METRIC_WINDOW, numerai_corr
from .store import EvaluationStore

AGGREGATION_ACTIVE_EQUAL_MEAN = "active_equal_mean"
AGGREGATION_CORR_WEIGHTED_MEAN = "corr_weighted_mean"
DEFAULT_PRIMARY_AGGREGATION_KIND = AGGREGATION_CORR_WEIGHTED_MEAN
DEFAULT_AGGREGATION_KINDS = (
    DEFAULT_PRIMARY_AGGREGATION_KIND,
    AGGREGATION_ACTIVE_EQUAL_MEAN,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _positive_corr_weight(corr: float) -> float:
    return max(float(corr), 0.0)


def _lagged_corr_weight(
    store: EvaluationStore,
    *,
    signal_id: str,
    subject_id: str,
    asset: str,
    target_id: str,
    evaluation_id: str,
    window_size: int,
) -> float:
    rows = store.conn.execute(
        """
        SELECT prediction_value, observation_value
        FROM evaluation_snapshots
        WHERE signal_id = ? AND subject_id = ? AND target_id = ? AND evaluation_id < ?
        ORDER BY evaluation_id DESC
        LIMIT ?
        """,
        (signal_id, subject_id, target_id, evaluation_id, int(window_size)),
    ).fetchall()
    if not rows:
        return 0.0

    rows = list(reversed(rows))
    predictions = pd.Series(
        [float(row["prediction_value"]) for row in rows],
        index=range(len(rows)),
        dtype=float,
    )
    observations = pd.Series(
        [float(row["observation_value"]) for row in rows],
        index=range(len(rows)),
        dtype=float,
    )
    return _positive_corr_weight(numerai_corr(predictions, observations))


def refresh_target_meta_predictions(
    store: EvaluationStore,
    *,
    subject_id: str | None = None,
    asset: str | None = None,
    target_id: str = DEFAULT_TARGET,
    aggregation_kinds: tuple[str, ...] = DEFAULT_AGGREGATION_KINDS,
    recorded_at: str | None = None,
    window_size: int = DEFAULT_METRIC_WINDOW,
) -> None:
    resolved_subject_id = DEFAULT_SUBJECT_ID if subject_id is None else subject_id
    resolved_asset = default_runtime_asset(resolved_subject_id) if asset is None else asset
    signals = store.list_signals(
        subject_id=resolved_subject_id,
        asset=resolved_asset,
        target_id=target_id,
    )
    if not signals:
        return

    rows = store.conn.execute(
        """
        SELECT p.evaluation_id, p.signal_id, p.value
        FROM predictions AS p
        JOIN signals AS h ON h.signal_id = p.signal_id
        WHERE h.subject_id = ? AND h.target_id = ?
        ORDER BY p.evaluation_id ASC, p.signal_id ASC
        """,
        (resolved_subject_id, target_id),
    ).fetchall()
    if not rows:
        return

    grouped: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        evaluation_id = str(row["evaluation_id"])
        grouped.setdefault(evaluation_id, []).append(
            (str(row["signal_id"]), float(row["value"]))
        )

    timestamp = recorded_at or _utc_now()
    for evaluation_id, contributors in grouped.items():
        for aggregation_kind in aggregation_kinds:
            if aggregation_kind == AGGREGATION_ACTIVE_EQUAL_MEAN:
                weights = {
                    signal_id: 1.0
                    for signal_id, _ in contributors
                }
            elif aggregation_kind == AGGREGATION_CORR_WEIGHTED_MEAN:
                weights = {
                    signal_id: _lagged_corr_weight(
                        store,
                        signal_id=signal_id,
                        subject_id=resolved_subject_id,
                        asset=resolved_asset,
                        target_id=target_id,
                        evaluation_id=evaluation_id,
                        window_size=window_size,
                    )
                    for signal_id, _ in contributors
                }
            else:
                raise ValueError(f"unknown aggregation kind: {aggregation_kind}")

            total_weight = sum(weights.values())
            if total_weight <= 0.0:
                weights = {
                    signal_id: 1.0
                    for signal_id, _ in contributors
                }
                total_weight = float(len(contributors))

            normalized_weights = {
                signal_id: weight / total_weight
                for signal_id, weight in weights.items()
            }
            value = sum(
                prediction * normalized_weights[signal_id]
                for signal_id, prediction in contributors
            )
            details_json = json.dumps(
                {
                    "contributors": [
                        {
                            "signal_id": signal_id,
                            "prediction": prediction,
                            "weight": normalized_weights[signal_id],
                            "weight_source": (
                                "equal"
                                if aggregation_kind == AGGREGATION_ACTIVE_EQUAL_MEAN
                                else "lagged_corr"
                            ),
                        }
                        for signal_id, prediction in contributors
                    ]
                },
                sort_keys=True,
            )
            store.upsert_meta_prediction(
                evaluation_id=evaluation_id,
                subject_id=resolved_subject_id,
                asset=resolved_asset,
                target_id=target_id,
                aggregation_kind=aggregation_kind,
                value=value,
                contributor_count=len(contributors),
                details_json=details_json,
                recorded_at=timestamp,
            )
