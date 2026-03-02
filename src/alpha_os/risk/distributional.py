"""Distribution-aware risk utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DistributionStats:
    ready: bool
    sample_size: int
    mean: float
    std: float
    left_tail_prob: float
    cvar: float


def estimate_distribution(
    returns: np.ndarray,
    window: int = 63,
    min_samples: int = 20,
    tail_sigma: float = 2.0,
    cvar_alpha: float = 0.05,
) -> DistributionStats:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if window > 0 and len(arr) > window:
        arr = arr[-window:]

    n = int(len(arr))
    if n < min_samples:
        return DistributionStats(
            ready=False,
            sample_size=n,
            mean=0.0,
            std=0.0,
            left_tail_prob=0.0,
            cvar=0.0,
        )

    mean = float(np.mean(arr))
    std = float(np.std(arr))
    left_threshold = mean - tail_sigma * std
    left_tail_prob = float(np.mean(arr < left_threshold))

    alpha = float(np.clip(cvar_alpha, 1e-6, 1.0))
    n_tail = max(1, int(np.ceil(alpha * n)))
    tail = np.partition(arr, n_tail - 1)[:n_tail]
    cvar = float(np.mean(tail))

    return DistributionStats(
        ready=True,
        sample_size=n,
        mean=mean,
        std=std,
        left_tail_prob=left_tail_prob,
        cvar=cvar,
    )


def passes_distributional_gate(
    stats: DistributionStats,
    max_left_tail_prob: float,
    max_cvar_abs: float,
) -> bool:
    if not stats.ready:
        return True
    return (
        stats.left_tail_prob <= max_left_tail_prob
        and abs(min(stats.cvar, 0.0)) <= max_cvar_abs
    )


def kelly_scale(
    stats: DistributionStats,
    kelly_fraction: float = 0.5,
    max_leverage: float = 1.0,
) -> float:
    if not stats.ready:
        return 1.0
    var = stats.std * stats.std
    if var <= 1e-12:
        return 1.0
    raw = kelly_fraction * (stats.mean / var)
    return float(np.clip(raw, 0.0, max_leverage))
