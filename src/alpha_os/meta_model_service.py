from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from .config import DEFAULT_ASSET, DEFAULT_TARGET
from .scoring import DEFAULT_METRIC_WINDOW, numerai_corr
from .store import EvaluationStore

AGGREGATION_ACTIVE_EQUAL_MEAN = "active_equal_mean"
AGGREGATION_CORR_WEIGHTED_MEAN = "corr_weighted_mean"
DEFAULT_AGGREGATION_KINDS = (
    AGGREGATION_ACTIVE_EQUAL_MEAN,
    AGGREGATION_CORR_WEIGHTED_MEAN,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _positive_corr_weight(corr: float) -> float:
    return max(float(corr), 0.0)


def _lagged_corr_weight(
    store: EvaluationStore,
    *,
    hypothesis_id: str,
    asset: str,
    target_id: str,
    evaluation_id: str,
    window_size: int,
) -> float:
    rows = store.conn.execute(
        """
        SELECT prediction_value, observation_value
        FROM evaluation_snapshots
        WHERE hypothesis_id = ? AND asset = ? AND target_id = ? AND evaluation_id < ?
        ORDER BY evaluation_id DESC
        LIMIT ?
        """,
        (hypothesis_id, asset, target_id, evaluation_id, int(window_size)),
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
    asset: str = DEFAULT_ASSET,
    target_id: str = DEFAULT_TARGET,
    aggregation_kinds: tuple[str, ...] = DEFAULT_AGGREGATION_KINDS,
    recorded_at: str | None = None,
    window_size: int = DEFAULT_METRIC_WINDOW,
) -> None:
    active_hypotheses = store.list_hypotheses(asset=asset, target_id=target_id)
    active_hypotheses = [item for item in active_hypotheses if item.status == "active"]
    if not active_hypotheses:
        return

    rows = store.conn.execute(
        """
        SELECT p.evaluation_id, p.hypothesis_id, p.value
        FROM predictions AS p
        JOIN hypotheses AS h ON h.hypothesis_id = p.hypothesis_id
        WHERE h.asset = ? AND h.target_id = ? AND h.status = 'active'
        ORDER BY p.evaluation_id ASC, p.hypothesis_id ASC
        """,
        (asset, target_id),
    ).fetchall()
    if not rows:
        return

    grouped: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        evaluation_id = str(row["evaluation_id"])
        grouped.setdefault(evaluation_id, []).append(
            (str(row["hypothesis_id"]), float(row["value"]))
        )

    timestamp = recorded_at or _utc_now()
    for evaluation_id, contributors in grouped.items():
        for aggregation_kind in aggregation_kinds:
            if aggregation_kind == AGGREGATION_ACTIVE_EQUAL_MEAN:
                weights = {hypothesis_id: 1.0 for hypothesis_id, _ in contributors}
            elif aggregation_kind == AGGREGATION_CORR_WEIGHTED_MEAN:
                weights = {
                    hypothesis_id: _lagged_corr_weight(
                        store,
                        hypothesis_id=hypothesis_id,
                        asset=asset,
                        target_id=target_id,
                        evaluation_id=evaluation_id,
                        window_size=window_size,
                    )
                    for hypothesis_id, _ in contributors
                }
            else:
                raise ValueError(f"unknown aggregation kind: {aggregation_kind}")

            total_weight = sum(weights.values())
            if total_weight <= 0.0:
                weights = {hypothesis_id: 1.0 for hypothesis_id, _ in contributors}
                total_weight = float(len(contributors))

            normalized_weights = {
                hypothesis_id: weight / total_weight
                for hypothesis_id, weight in weights.items()
            }
            value = sum(
                prediction * normalized_weights[hypothesis_id]
                for hypothesis_id, prediction in contributors
            )
            details_json = json.dumps(
                {
                    "contributors": [
                        {
                            "hypothesis_id": hypothesis_id,
                            "prediction": prediction,
                            "weight": normalized_weights[hypothesis_id],
                            "weight_source": (
                                "equal"
                                if aggregation_kind == AGGREGATION_ACTIVE_EQUAL_MEAN
                                else "lagged_corr"
                            ),
                        }
                        for hypothesis_id, prediction in contributors
                    ]
                },
                sort_keys=True,
            )
            store.upsert_meta_prediction(
                evaluation_id=evaluation_id,
                asset=asset,
                target_id=target_id,
                aggregation_kind=aggregation_kind,
                value=value,
                contributor_count=len(contributors),
                details_json=details_json,
                recorded_at=timestamp,
            )


def refresh_target_meta_prediction_metrics(
    store: EvaluationStore,
    *,
    asset: str = DEFAULT_ASSET,
    target_id: str = DEFAULT_TARGET,
    aggregation_kinds: tuple[str, ...] = DEFAULT_AGGREGATION_KINDS,
    recorded_at: str | None = None,
    window_size: int = DEFAULT_METRIC_WINDOW,
) -> None:
    timestamp = recorded_at or _utc_now()
    for aggregation_kind in aggregation_kinds:
        rows = store.conn.execute(
            """
            SELECT mp.evaluation_id, mp.value AS prediction_value, o.value AS observation_value
            FROM meta_predictions AS mp
            JOIN observations AS o ON o.evaluation_id = mp.evaluation_id
            WHERE mp.asset = ? AND mp.target_id = ? AND mp.aggregation_kind = ?
            ORDER BY mp.evaluation_id DESC
            LIMIT ?
            """,
            (asset, target_id, aggregation_kind, int(window_size)),
        ).fetchall()
        if not rows:
            continue

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
        corr = numerai_corr(predictions, observations)
        store.upsert_meta_prediction_metric(
            aggregation_kind=aggregation_kind,
            asset=asset,
            target_id=target_id,
            corr=corr,
            sample_count=len(predictions),
            window_size=int(window_size),
            start_evaluation_id=evaluation_ids[0],
            end_evaluation_id=evaluation_ids[-1],
            recorded_at=timestamp,
        )
