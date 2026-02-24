"""Alpha combiner — select low-correlation alphas and build equal-weight portfolio."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CombinerConfig:
    max_correlation: float = 0.3
    max_alphas: int = 30


def select_low_correlation(
    signals: np.ndarray,
    sharpes: np.ndarray,
    config: CombinerConfig | None = None,
) -> list[int]:
    """Greedy forward selection of low-correlation alphas.

    Parameters
    ----------
    signals : (n_alphas, n_days) signal matrix
    sharpes : (n_alphas,) Sharpe ratios for ranking
    config : combiner configuration

    Returns
    -------
    List of selected alpha indices.
    """
    cfg = config or CombinerConfig()
    n = signals.shape[0]
    if n == 0:
        return []

    # Rank by Sharpe descending
    order = np.argsort(-sharpes)
    selected: list[int] = [int(order[0])]

    for idx in order[1:]:
        if len(selected) >= cfg.max_alphas:
            break
        idx = int(idx)
        sig = signals[idx]
        sig_clean = np.nan_to_num(sig)

        # Check correlation with all selected
        too_correlated = False
        for sel_idx in selected:
            sel_sig = np.nan_to_num(signals[sel_idx])
            n_pts = min(len(sig_clean), len(sel_sig))
            if n_pts < 10:
                continue
            corr = np.corrcoef(sig_clean[:n_pts], sel_sig[:n_pts])[0, 1]
            if np.isnan(corr):
                continue
            if abs(corr) > cfg.max_correlation:
                too_correlated = True
                break

        if not too_correlated:
            selected.append(idx)

    return selected


def equal_weight_combine(signals: np.ndarray, indices: list[int]) -> np.ndarray:
    """Combine selected signals with equal weight.

    Returns normalized combined signal clipped to [-1, 1].
    """
    if not indices:
        return np.zeros(signals.shape[1])

    selected = signals[indices]
    # Replace NaN with 0 before combining
    selected = np.nan_to_num(selected)
    combined = selected.mean(axis=0)

    std = combined.std()
    if std > 0:
        combined = combined / std
    return np.clip(combined, -1, 1)
