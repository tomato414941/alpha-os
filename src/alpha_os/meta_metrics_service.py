from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from .config import DEFAULT_ASSET, DEFAULT_TARGET
from .meta_aggregation_service import DEFAULT_AGGREGATION_KINDS
from .scoring import DEFAULT_METRIC_WINDOW, numerai_corr
from .store import EvaluationStore


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


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
