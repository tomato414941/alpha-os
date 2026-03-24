from __future__ import annotations

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
        if bool(record.metadata.get("lifecycle_capital_eligible", record.stake > 0))
        and record.stake > 0
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
