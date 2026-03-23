from __future__ import annotations

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
    tradable = [record for record in records if record.stake > 0]
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


def _rank_key(
    estimate: QualityEstimate,
    prior_quality: float,
) -> tuple[float, float, float]:
    return (
        estimate.blended_quality,
        estimate.confidence,
        prior_quality,
    )
