from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from .config import DEFAULT_SUBJECT_ID, DEFAULT_TARGET, default_runtime_asset
from .scoring import DEFAULT_METRIC_WINDOW, compute_signal_metrics
from .store import EvaluationStore

MMC_BASELINE_PEER_MEAN = "peer_mean"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def refresh_signal_metrics(
    store: EvaluationStore,
    *,
    signal_id: str,
    subject_id: str | None = None,
    asset: str | None = None,
    target_id: str = DEFAULT_TARGET,
    recorded_at: str | None = None,
    window_size: int = DEFAULT_METRIC_WINDOW,
) -> None:
    resolved_subject_id = DEFAULT_SUBJECT_ID if subject_id is None else subject_id
    rows = store.conn.execute(
        """
        SELECT p.evaluation_id, p.value AS prediction_value, o.value AS observation_value
        FROM predictions AS p
        JOIN observations AS o ON o.evaluation_id = p.evaluation_id
        JOIN signals AS h ON h.signal_id = p.signal_id
        WHERE p.signal_id = ? AND h.subject_id = ? AND h.target_id = ?
        ORDER BY p.evaluation_id DESC
        LIMIT ?
        """,
        (
            signal_id,
            resolved_subject_id,
            target_id,
            int(window_size),
        ),
    ).fetchall()

    effective_recorded_at = recorded_at or _utc_now()
    if not rows:
        with store.conn:
            store.conn.execute(
                """
                INSERT INTO signal_metrics (
                    signal_id, corr, mmc, mmc_baseline_type, mmc_peer_count,
                    sample_count, mmc_sample_count, window_size,
                    start_evaluation_id, end_evaluation_id, updated_at
                )
                VALUES (?, 0.0, NULL, NULL, 0, 0, 0, ?, NULL, NULL, ?)
                ON CONFLICT(signal_id) DO UPDATE SET
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
                (signal_id, int(window_size), effective_recorded_at),
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
        SELECT COUNT(DISTINCT p.signal_id)
        FROM predictions AS p
        JOIN signals AS h ON h.signal_id = p.signal_id
        WHERE p.evaluation_id IN ({placeholders})
          AND h.subject_id = ?
          AND h.target_id = ?
          AND p.signal_id <> ?
        """,
        tuple(evaluation_ids) + (resolved_subject_id, target_id, signal_id),
    ).fetchone()[0]
    peer_rows = store.conn.execute(
        f"""
        SELECT p.evaluation_id, AVG(p.value) AS meta_prediction
        FROM predictions AS p
        JOIN signals AS h ON h.signal_id = p.signal_id
        WHERE p.evaluation_id IN ({placeholders})
          AND h.subject_id = ?
          AND h.target_id = ?
          AND p.signal_id <> ?
        GROUP BY p.evaluation_id
        ORDER BY p.evaluation_id ASC
        """,
        tuple(evaluation_ids) + (resolved_subject_id, target_id, signal_id),
    ).fetchall()
    if peer_rows:
        meta_model = pd.Series(
            [float(row["meta_prediction"]) for row in peer_rows],
            index=[str(row["evaluation_id"]) for row in peer_rows],
            dtype=float,
        )

    metrics = compute_signal_metrics(
        predictions=predictions,
        target=observations,
        meta_model=meta_model,
        window_size=window_size,
    )
    with store.conn:
        store.conn.execute(
            """
            INSERT INTO signal_metrics (
                signal_id, corr, mmc, mmc_baseline_type, mmc_peer_count,
                sample_count, mmc_sample_count, window_size,
                start_evaluation_id, end_evaluation_id, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signal_id) DO UPDATE SET
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
                signal_id,
                metrics.corr,
                metrics.mmc,
                MMC_BASELINE_PEER_MEAN,
                int(peer_count),
                metrics.sample_count,
                metrics.mmc_sample_count,
                metrics.window_size,
                evaluation_ids[0],
                evaluation_ids[-1],
                effective_recorded_at,
            ),
        )


def refresh_target_metrics(
    store: EvaluationStore,
    *,
    subject_id: str | None = None,
    asset: str | None = None,
    target_id: str = DEFAULT_TARGET,
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
    for signal in signals:
        if (
            signal.prediction_count <= 0
            and signal.observation_count <= 0
        ):
            continue
        refresh_signal_metrics(
            store,
            signal_id=signal.signal_id,
            subject_id=signal.subject_id,
            asset=resolved_asset,
            target_id=target_id,
            recorded_at=recorded_at,
            window_size=window_size,
        )
