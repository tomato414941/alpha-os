"""Blended alpha quality estimates for live selection and lifecycle."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class QualityEstimate:
    prior_quality: float
    live_quality: float
    blended_quality: float
    confidence: float
    n_observations: int
    has_min_observations: bool


def rolling_fitness(
    returns: list[float] | np.ndarray,
    *,
    metric: str = "sharpe",
    rolling_window: int = 63,
) -> float:
    """Compute rolling live fitness from the available forward returns."""
    if len(returns) == 0:
        return 0.0

    recent = np.asarray(returns[-rolling_window:], dtype=np.float64)
    if recent.size == 0:
        return 0.0

    if metric == "sharpe":
        std = float(recent.std())
        if std <= 1e-12:
            return 0.0
        return float(recent.mean() / std * sqrt(252))

    if metric == "log_growth":
        clipped = np.clip(recent, -0.999999, None)
        return float(np.mean(np.log1p(clipped)) * 252)

    raise ValueError(f"Unsupported fitness metric: {metric}")


def blend_quality(
    prior_quality: float,
    returns: list[float] | np.ndarray,
    *,
    metric: str = "sharpe",
    rolling_window: int = 63,
    min_observations: int = 20,
    full_weight_observations: int = 63,
) -> QualityEstimate:
    """Blend historical OOS quality with live forward quality.

    Confidence rises linearly with the number of forward observations.
    With no live observations, the estimate falls back to the historical prior.
    """
    n_observations = len(returns)
    live_quality = rolling_fitness(
        returns,
        metric=metric,
        rolling_window=rolling_window,
    )

    denom = max(full_weight_observations, 1)
    confidence = min(max(n_observations, 0) / denom, 1.0)
    blended_quality = (1.0 - confidence) * prior_quality + confidence * live_quality

    return QualityEstimate(
        prior_quality=float(prior_quality),
        live_quality=float(live_quality),
        blended_quality=float(blended_quality),
        confidence=float(confidence),
        n_observations=int(n_observations),
        has_min_observations=n_observations >= min_observations,
    )
