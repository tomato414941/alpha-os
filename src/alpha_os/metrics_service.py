from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from .config import DEFAULT_ASSET, DEFAULT_TARGET
from .scoring import DEFAULT_METRIC_WINDOW, compute_hypothesis_metrics
from .store import EvaluationStore

MMC_BASELINE_ACTIVE_PEER_MEAN = "active_peer_mean"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def refresh_hypothesis_metrics(
    store: EvaluationStore,
    *,
    hypothesis_id: str,
    asset: str = DEFAULT_ASSET,
    target_id: str = DEFAULT_TARGET,
    recorded_at: str | None = None,
    window_size: int = DEFAULT_METRIC_WINDOW,
) -> None:
    rows = store.conn.execute(
        """
        SELECT p.evaluation_id, p.value AS prediction_value, o.value AS observation_value
        FROM predictions AS p
        JOIN observations AS o ON o.evaluation_id = p.evaluation_id
        JOIN hypotheses AS h ON h.hypothesis_id = p.hypothesis_id
        WHERE p.hypothesis_id = ? AND h.asset = ? AND h.target_id = ?
        ORDER BY p.evaluation_id DESC
        LIMIT ?
        """,
        (hypothesis_id, asset, target_id, int(window_size)),
    ).fetchall()

    effective_recorded_at = recorded_at or _utc_now()
    if not rows:
        store.conn.execute(
            """
            INSERT INTO hypothesis_metrics (
                hypothesis_id, corr, mmc, mmc_baseline_type, mmc_peer_count,
                sample_count, mmc_sample_count, window_size,
                start_evaluation_id, end_evaluation_id, updated_at
            )
            VALUES (?, 0.0, NULL, NULL, 0, 0, 0, ?, NULL, NULL, ?)
            ON CONFLICT(hypothesis_id) DO UPDATE SET
                corr = excluded.corr,
                mmc = excluded.mmc,
                mmc_baseline_type = excluded.mmc_baseline_type,
                mmc_peer_count = excluded.mmc_peer_count,
                sample_count = excluded.sample_count,
                mmc_sample_count = excluded.mmc_sample_count,
                window_size = excluded.window_size,
                start_evaluation_id = excluded.start_evaluation_id,
                end_evaluation_id = excluded.end_evaluation_id,
                updated_at = excluded.updated_at
            """,
            (hypothesis_id, int(window_size), effective_recorded_at),
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
    peer_count = store.conn.execute(
        f"""
        SELECT COUNT(DISTINCT p.hypothesis_id)
        FROM predictions AS p
        JOIN hypotheses AS h ON h.hypothesis_id = p.hypothesis_id
        WHERE p.evaluation_id IN ({placeholders})
          AND h.asset = ?
          AND h.target_id = ?
          AND h.status = 'active'
          AND p.hypothesis_id <> ?
        """,
        tuple(evaluation_ids) + (asset, target_id, hypothesis_id),
    ).fetchone()[0]
    peer_rows = store.conn.execute(
        f"""
        SELECT p.evaluation_id, AVG(p.value) AS meta_prediction
        FROM predictions AS p
        JOIN hypotheses AS h ON h.hypothesis_id = p.hypothesis_id
        WHERE p.evaluation_id IN ({placeholders})
          AND h.asset = ?
          AND h.target_id = ?
          AND h.status = 'active'
          AND p.hypothesis_id <> ?
        GROUP BY p.evaluation_id
        ORDER BY p.evaluation_id ASC
        """,
        tuple(evaluation_ids) + (asset, target_id, hypothesis_id),
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
    store.conn.execute(
        """
        INSERT INTO hypothesis_metrics (
            hypothesis_id, corr, mmc, mmc_baseline_type, mmc_peer_count,
            sample_count, mmc_sample_count, window_size,
            start_evaluation_id, end_evaluation_id, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(hypothesis_id) DO UPDATE SET
            corr = excluded.corr,
            mmc = excluded.mmc,
            mmc_baseline_type = excluded.mmc_baseline_type,
            mmc_peer_count = excluded.mmc_peer_count,
            sample_count = excluded.sample_count,
            mmc_sample_count = excluded.mmc_sample_count,
            window_size = excluded.window_size,
            start_evaluation_id = excluded.start_evaluation_id,
            end_evaluation_id = excluded.end_evaluation_id,
            updated_at = excluded.updated_at
        """,
        (
            hypothesis_id,
            metrics.corr,
            metrics.mmc,
            MMC_BASELINE_ACTIVE_PEER_MEAN,
            int(peer_count),
            metrics.sample_count,
            metrics.mmc_sample_count,
            metrics.window_size,
            evaluation_ids[0],
            evaluation_ids[-1],
            effective_recorded_at,
        ),
    )
