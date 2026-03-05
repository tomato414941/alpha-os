"""Voter scoring — weight alphas by recency and recent accuracy."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..forward.tracker import ForwardRecord


def recency_weight(
    ages_days: np.ndarray,
    half_life: float = 2.0,
) -> np.ndarray:
    """Exponential decay weight by alpha age.

    Newer alphas get higher weight. At half_life days, weight = 0.5.

    Parameters
    ----------
    ages_days : (N,) age of each alpha in days
    half_life : days for weight to halve

    Returns
    -------
    weights : (N,) normalized weights summing to 1.0
    """
    if len(ages_days) == 0:
        return np.array([], dtype=np.float64)
    raw = 0.5 ** (ages_days / max(half_life, 1e-12))
    total = raw.sum()
    if total > 0:
        return raw / total
    return np.full(len(ages_days), 1.0 / len(ages_days))


def accuracy_weight(
    signals: np.ndarray,
    outcomes: np.ndarray,
    lookback: int = 5,
) -> np.ndarray:
    """Weight by directional accuracy over recent days.

    Parameters
    ----------
    signals : (N, T) signal history
    outcomes : (T,) actual returns
    lookback : number of recent days to evaluate

    Returns
    -------
    accuracy : (N,) fraction of correct direction calls in [0, 1]
    """
    N, T = signals.shape
    if T == 0 or lookback <= 0:
        return np.full(N, 0.5)
    L = min(lookback, T)
    sig = signals[:, -L:]
    out = outcomes[-L:]
    correct = (np.sign(sig) == np.sign(out)).astype(np.float64)
    return correct.mean(axis=1)


def accuracy_from_forward(
    records: list[ForwardRecord],
    lookback: int = 5,
) -> float:
    """Directional accuracy from forward tracker records.

    Parameters
    ----------
    records : per-alpha ForwardRecord list (ordered by date)
    lookback : number of recent records to evaluate

    Returns
    -------
    Fraction of correct direction calls in [0, 1]. Returns 0.5 if no data.
    """
    recent = records[-lookback:] if records else []
    if not recent:
        return 0.5
    correct = sum(
        1 for r in recent
        if np.sign(r.signal_value) == np.sign(r.daily_return)
    )
    return correct / len(recent)
