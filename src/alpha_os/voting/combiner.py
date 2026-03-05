"""Vote combiner — bridge between trader data and voting aggregator."""
from __future__ import annotations

import time

import numpy as np

from ..alpha.registry import AlphaRecord
from ..forward.tracker import ForwardTracker
from .aggregator import VoteResult, vote_aggregate
from .scorer import accuracy_from_forward, recency_weight


def vote_combine(
    alpha_signals: dict[str, float],
    forward_tracker: ForwardTracker,
    registry_records: dict[str, AlphaRecord],
    recency_half_life: float = 2.0,
    accuracy_lookback: int = 5,
    min_voters: int = 5,
) -> VoteResult:
    """Build voter weights from recency × accuracy and aggregate.

    Parameters
    ----------
    alpha_signals : {alpha_id: signal_value} current signals
    forward_tracker : for per-alpha historical records
    registry_records : {alpha_id: AlphaRecord} for created_at
    recency_half_life : half-life in days for recency weighting
    accuracy_lookback : days of forward records for accuracy
    min_voters : minimum alphas for valid vote

    Returns
    -------
    VoteResult with direction, confidence, and diagnostics
    """
    if not alpha_signals:
        return VoteResult(
            direction=0.0, confidence=0.0,
            n_voters=0, long_pct=0.0, short_pct=0.0,
        )

    alpha_ids = list(alpha_signals.keys())
    now = time.time()

    # Compute per-alpha age and accuracy
    ages = np.array([
        max(0.0, (now - registry_records[aid].created_at) / 86400.0)
        if aid in registry_records else 0.0
        for aid in alpha_ids
    ])
    accuracies = np.array([
        accuracy_from_forward(
            forward_tracker.get_records(aid),
            lookback=accuracy_lookback,
        )
        for aid in alpha_ids
    ])

    # Recency weights (normalized)
    rec_w = recency_weight(ages, half_life=recency_half_life)

    # Combined weight = recency × accuracy (then normalize)
    raw_weights = rec_w * accuracies
    total = raw_weights.sum()
    if total > 0:
        weights = raw_weights / total
    else:
        weights = np.full(len(alpha_ids), 1.0 / len(alpha_ids))

    weights_dict = {aid: float(weights[i]) for i, aid in enumerate(alpha_ids)}

    return vote_aggregate(alpha_signals, weights_dict, min_voters=min_voters)
