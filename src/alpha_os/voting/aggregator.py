"""Vote aggregation — combine ephemeral alpha signals into a single vote."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VoteResult:
    """Result of alpha voting."""
    direction: float     # sign of weighted vote mean
    confidence: float    # agreement level [0, 1]
    n_voters: int        # number of participating alphas
    long_pct: float      # fraction voting long
    short_pct: float     # fraction voting short


def vote_aggregate(
    signals: dict[str, float],
    weights: dict[str, float],
    min_voters: int = 5,
) -> VoteResult:
    """Aggregate alpha signals as weighted votes.

    Each alpha casts a directional vote (sign of signal) with magnitude
    as conviction. The aggregate is a weighted tally.

    Parameters
    ----------
    signals : {alpha_id: signal_value} current signals
    weights : {alpha_id: weight} voter weights (e.g. recency × accuracy)
    min_voters : minimum alphas required for a valid vote

    Returns
    -------
    VoteResult with direction, confidence, and diagnostics
    """
    if len(signals) < min_voters:
        return VoteResult(
            direction=0.0, confidence=0.0,
            n_voters=len(signals), long_pct=0.0, short_pct=0.0,
        )

    ids = list(signals.keys())
    s = np.array([signals[a] for a in ids])
    w = np.array([weights.get(a, 0.0) for a in ids])
    w_sum = w.sum()
    if w_sum <= 0:
        w = np.ones(len(ids)) / len(ids)
        w_sum = 1.0
    w_norm = w / w_sum

    # Weighted mean signal
    mean = float(np.dot(w_norm, s))
    # Directional tally
    long_mask = s > 0
    short_mask = s < 0
    long_pct = float(w_norm[long_mask].sum()) if long_mask.any() else 0.0
    short_pct = float(w_norm[short_mask].sum()) if short_mask.any() else 0.0
    # Confidence: how lopsided the vote is
    confidence = abs(long_pct - short_pct)

    return VoteResult(
        direction=float(np.sign(mean)) if abs(mean) > 1e-12 else 0.0,
        confidence=confidence,
        n_voters=len(ids),
        long_pct=long_pct,
        short_pct=short_pct,
    )
