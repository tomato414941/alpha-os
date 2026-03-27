from __future__ import annotations

import numpy as np

from .breadth import hypothesis_signal_series
from .combiner import CombinerConfig, select_low_correlation
from .state_lifecycle import ST_ACTIVE, ST_DORMANT
from .quality import QualityEstimate


def trading_candidate_limit(max_trading: int) -> int:
    return max(max_trading, 1) * 5


def rank_trading_records(
    records: list,
    estimate_for,
    *,
    max_trading: int,
    shortlist_preselect_factor: int,
    metric: str,
) -> list:
    n_candidates = trading_candidate_limit(max_trading)
    preselect_n = max(
        n_candidates,
        max_trading * max(shortlist_preselect_factor, 1),
    )
    tradable = [
        record
        for record in records
        if bool(
            record.metadata.get(
                "lifecycle_capital_backed",
                bool(record.metadata.get("lifecycle_capital_eligible", record.stake > 0))
                and record.stake > 0,
            )
        )
    ]
    tradable.sort(key=lambda record: record.stake, reverse=True)
    preselected = tradable[:preselect_n]
    preselected.sort(
        key=lambda record: _rank_key(
            estimate_for(record),
            record.oos_fitness(metric),
        ),
        reverse=True,
    )
    return preselected[:n_candidates]


def rank_trading_indices(
    records: list,
    state_codes,
    *,
    prior_quality,
    blended_quality,
    confidence,
    max_trading: int,
    metric: str,
    shortlist_preselect_factor: int,
) -> list[int]:
    n_candidates = trading_candidate_limit(max_trading)
    preselect_n = max(
        n_candidates,
        max_trading * max(shortlist_preselect_factor, 1),
    )
    tradable = [idx for idx, state in enumerate(state_codes) if state == ST_ACTIVE]
    tradable.sort(key=lambda idx: prior_quality[idx], reverse=True)
    preselected = tradable[:preselect_n]
    preselected.sort(
        key=lambda idx: (
            blended_quality[idx],
            confidence[idx],
            prior_quality[idx],
            records[idx].oos_fitness(metric),
        ),
        reverse=True,
    )
    return preselected[:n_candidates]


def select_decorrelated_trading_ids(
    candidate_ids: list[str],
    *,
    signal_history_by_id: dict[str, np.ndarray],
    blended_quality_by_id: dict[str, float],
    max_trading: int,
    max_correlation: float | None = None,
) -> list[str]:
    if len(candidate_ids) <= max_trading:
        return list(candidate_ids)

    filtered_candidate_ids: list[str] = []
    signal_histories: list[np.ndarray] = []
    for hypothesis_id in candidate_ids:
        history = signal_history_by_id.get(hypothesis_id)
        if history is None:
            continue
        signal_histories.append(np.asarray(history, dtype=np.float64))
        filtered_candidate_ids.append(hypothesis_id)

    if len(filtered_candidate_ids) <= max_trading:
        return filtered_candidate_ids

    quality_scores = np.array(
        [blended_quality_by_id[hypothesis_id] for hypothesis_id in filtered_candidate_ids],
        dtype=np.float64,
    )
    config = CombinerConfig(
        max_alphas=max_trading,
        max_correlation=CombinerConfig().max_correlation if max_correlation is None else max_correlation,
    )
    selected_idx = select_low_correlation(
        np.array(signal_histories),
        quality_scores,
        config,
    )
    return [filtered_candidate_ids[idx] for idx in selected_idx]


def build_trading_signal_history_map(
    records: list,
    *,
    candidate_ids: list[str],
    data: dict[str, np.ndarray],
    asset: str,
    fallback_signal_arrays: dict[str, np.ndarray],
    lookback: int,
) -> dict[str, np.ndarray]:
    record_map = {record.hypothesis_id: record for record in records}
    signal_history_by_id: dict[str, np.ndarray] = {}
    for hypothesis_id in candidate_ids:
        record = record_map.get(hypothesis_id)
        history = None
        if record is not None:
            history = hypothesis_signal_series(record, data=data, asset=asset)
        if history is None:
            history = fallback_signal_arrays.get(hypothesis_id)
        if history is None:
            continue
        signal_history_by_id[hypothesis_id] = np.asarray(history[-lookback:], dtype=np.float64)
    return signal_history_by_id


def dormant_indices(state_codes) -> list[int]:
    return [idx for idx, state in enumerate(state_codes) if state == ST_DORMANT]


def _rank_key(
    estimate: QualityEstimate,
    prior_quality: float,
) -> tuple[float, float, float]:
    return (
        estimate.blended_quality,
        estimate.confidence,
        prior_quality,
    )
