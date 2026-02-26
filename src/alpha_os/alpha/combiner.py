"""Alpha combiner — signal combination with quality × diversity weighting."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Weighted combiner — quality × diversity, all alphas participate
# ---------------------------------------------------------------------------


@dataclass
class WeightedCombinerConfig:
    min_weight: float = 1e-4
    chunk_size: int = 1000
    diversity_recompute_days: int = 63
    corr_lookback: int = 252


def compute_diversity_scores(
    signals: np.ndarray,
    chunk_size: int = 1000,
) -> np.ndarray:
    """Compute diversity = 1 - mean(|corr(i,j)|) for each alpha.

    Uses chunked matrix multiplication to keep memory bounded.

    Parameters
    ----------
    signals : (N, T) signal matrix, should be NaN-free
    chunk_size : rows per chunk to limit peak memory

    Returns
    -------
    diversity : (N,) array in [0, 1]. Higher = more unique.
    """
    N, T = signals.shape
    if N <= 1:
        return np.ones(N, dtype=np.float64)

    # Standardize for Pearson correlation: z = (x - mean) / std
    means = signals.mean(axis=1, keepdims=True)
    stds = signals.std(axis=1, keepdims=True)
    stds = np.where(stds > 1e-12, stds, 1.0)
    z = (signals - means) / stds  # (N, T)

    abs_corr_sum = np.zeros(N, dtype=np.float64)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        # (chunk, T) @ (T, N) -> (chunk, N) correlation block
        chunk_corr = z[start:end] @ z.T / T
        np.abs(chunk_corr, out=chunk_corr)
        # Sum |corr| across all j, subtract self-correlation (1.0)
        abs_corr_sum[start:end] = chunk_corr.sum(axis=1) - 1.0

    avg_abs_corr = abs_corr_sum / max(N - 1, 1)
    return np.clip(1.0 - avg_abs_corr, 0.0, 1.0)


def compute_weights(
    sharpes: np.ndarray,
    diversity: np.ndarray,
    min_weight: float = 1e-4,
) -> np.ndarray:
    """Compute normalized weights = quality × diversity + min_weight.

    Parameters
    ----------
    sharpes : (N,) rolling Sharpe ratios (quality signal)
    diversity : (N,) diversity scores in [0, 1]
    min_weight : floor so no alpha is fully excluded

    Returns
    -------
    weights : (N,) normalized weights summing to 1.0
    """
    quality = np.maximum(sharpes, 0.0)
    raw = quality * diversity + min_weight
    total = raw.sum()
    if total > 0:
        return raw / total
    return np.full(len(sharpes), 1.0 / max(len(sharpes), 1))


def weighted_combine(
    signals: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted combination of signals (matrix version).

    Parameters
    ----------
    signals : (N, T) signal matrix
    weights : (N,) normalized weights

    Returns
    -------
    combined : (T,) combined signal clipped to [-1, 1]
    """
    combined = weights @ np.nan_to_num(signals)
    return np.clip(combined, -1.0, 1.0)


def weighted_combine_scalar(
    signals: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Weighted combination for scalar signals (trader daily cycle).

    Parameters
    ----------
    signals : {alpha_id: signal_value}
    weights : {alpha_id: weight}

    Returns
    -------
    combined signal clipped to [-1, 1]
    """
    total = 0.0
    for alpha_id, sig in signals.items():
        total += weights.get(alpha_id, 0.0) * sig
    return float(np.clip(total, -1.0, 1.0))
