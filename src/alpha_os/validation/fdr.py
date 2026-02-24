"""False Discovery Rate control — Benjamini-Hochberg procedure.

Given a set of alpha p-values, identifies which strategies survive
FDR correction at a given threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FDRResult:
    n_tested: int
    n_rejected: int
    threshold: float
    rejected_indices: list[int]
    adjusted_pvalues: list[float]


def benjamini_hochberg(
    p_values: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> FDRResult:
    """Apply Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : raw p-values for each strategy
    alpha : desired FDR level

    Returns
    -------
    FDRResult with indices of strategies that survive correction.
    """
    pvals = np.asarray(p_values, dtype=float)
    n = len(pvals)
    if n == 0:
        return FDRResult(0, 0, alpha, [], [])

    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]

    # BH adjusted p-values
    adjusted = np.empty(n)
    adjusted[sorted_idx[-1]] = sorted_pvals[-1]
    for i in range(n - 2, -1, -1):
        bh_val = sorted_pvals[i] * n / (i + 1)
        adjusted[sorted_idx[i]] = min(bh_val, adjusted[sorted_idx[i + 1]])

    # Find threshold: largest k where p_(k) <= k/n * alpha
    bh_threshold = 0.0
    for k in range(n):
        critical = (k + 1) / n * alpha
        if sorted_pvals[k] <= critical:
            bh_threshold = critical

    rejected = [int(i) for i in range(n) if adjusted[i] <= alpha]

    return FDRResult(
        n_tested=n,
        n_rejected=len(rejected),
        threshold=bh_threshold,
        rejected_indices=rejected,
        adjusted_pvalues=adjusted.tolist(),
    )
