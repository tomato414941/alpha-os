"""Two-level ensemble aggregation for Path B (MAP-Elites + distributional voting).

Level 1: Per-cell sign voting — each alpha's signal is reduced to {-1, +1},
         and per-cell long_pct is computed.
Level 2: Cross-cell distributional sizing — the distribution of cell-level
         long_pcts determines direction, confidence, and skew adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evolution.archive import AlphaArchive


@dataclass
class EnsembleResult:
    direction: float
    confidence: float
    skew_adj: float
    n_cells: int
    mu_cells: float
    sigma_cells: float
    skew_cells: float


def ensemble_sizing(
    cell_long_pcts: list[float],
    skew_k: float = 0.5,
    min_cells: int = 5,
) -> EnsembleResult:
    """Compute sizing parameters from per-cell long percentages.

    Parameters
    ----------
    cell_long_pcts : long_pct for each occupied cell
    skew_k : skewness penalty coefficient
    min_cells : minimum cells required for valid result
    """
    n = len(cell_long_pcts)
    if n < min_cells:
        return EnsembleResult(
            direction=0.0, confidence=0.0, skew_adj=1.0,
            n_cells=n, mu_cells=0.5, sigma_cells=0.0, skew_cells=0.0,
        )

    arr = np.array(cell_long_pcts)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))

    if n >= 3 and sigma > 0:
        skew = float(np.mean(((arr - mu) / sigma) ** 3))
    else:
        skew = 0.0

    centered = abs(mu - 0.5) * 2  # [0, 1]
    if centered + sigma > 0:
        confidence = centered / (centered + sigma)
    else:
        confidence = 0.0

    skew_adj = float(np.clip(1.0 - abs(skew) * skew_k, 0.5, 1.0))
    direction = 1.0 if mu > 0.5 else (-1.0 if mu < 0.5 else 0.0)

    return EnsembleResult(
        direction=direction,
        confidence=confidence,
        skew_adj=skew_adj,
        n_cells=n,
        mu_cells=mu,
        sigma_cells=sigma,
        skew_cells=skew,
    )


def compute_cell_long_pcts(
    archive: AlphaArchive,
    signals: dict[tuple[int, ...], list[float]],
) -> list[float]:
    """Compute per-cell long_pct from cell-grouped signals.

    Parameters
    ----------
    archive : MAP-Elites archive (used for cell structure)
    signals : {cell_key: [signal_values]} mapping

    Returns
    -------
    List of long_pct values, one per occupied cell with signals.
    """
    pcts = []
    for cell_key, cell_signals in signals.items():
        if not cell_signals:
            continue
        signs = [1 if s > 0 else 0 for s in cell_signals]
        pcts.append(sum(signs) / len(signs))
    return pcts
