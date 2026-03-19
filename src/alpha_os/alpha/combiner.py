"""Alpha combiner — signal combination with TC (True Contribution) weighting."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import sqrt

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# True Contribution (TC) — Numerai-inspired ensemble contribution metric
# ---------------------------------------------------------------------------

_ANNUALIZE = sqrt(252)


def compute_tc_scores(
    signal_arrays: dict[str, np.ndarray],
    returns: np.ndarray,
    min_observations: int = 20,
) -> dict[str, float]:
    """Leave-one-out ensemble Sharpe improvement for each alpha.

    TC_i = Sharpe(ensemble) - Sharpe(ensemble without alpha i).
    Positive TC means the alpha improves the ensemble.

    Parameters
    ----------
    signal_arrays : {alpha_id: (T,) signal} for each alpha
    returns : (T,) asset returns
    min_observations : minimum data points required

    Returns
    -------
    {alpha_id: tc_score}
    """
    ids = list(signal_arrays.keys())
    n = len(ids)
    if n == 0:
        return {}
    if len(returns) < min_observations:
        return {aid: 0.0 for aid in ids}

    T = len(returns)
    signals = np.array([
        _sanitize_signal_array(signal_arrays[aid][-T:]) for aid in ids
    ])

    # Equal-weight ensemble return
    ens = signals.mean(axis=0) * returns
    full_sharpe = _sharpe(ens)

    scores: dict[str, float] = {}
    for i, aid in enumerate(ids):
        if n == 1:
            scores[aid] = full_sharpe
            continue
        # Remove alpha i, equal weight the rest
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        ens_without = signals[mask].mean(axis=0) * returns
        scores[aid] = full_sharpe - _sharpe(ens_without)
    return scores


def compute_tc_weights(
    tc_scores: dict[str, float],
    min_weight: float = 1e-4,
    max_weight: float = 0.3,
) -> dict[str, float]:
    """Normalize TC scores to portfolio weights.

    Negative TC alphas get min_weight (kept for monitoring, not excluded).
    """
    if not tc_scores:
        return {}
    ids = list(tc_scores.keys())
    raw = np.array([max(tc_scores[aid], 0.0) + min_weight for aid in ids])
    # Cap individual weight
    raw = np.minimum(raw, max_weight * raw.sum())
    total = raw.sum()
    if total <= 0:
        eq = 1.0 / len(ids)
        return {aid: eq for aid in ids}
    weights = raw / total
    return {aid: float(weights[i]) for i, aid in enumerate(ids)}


def _sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe of a return series."""
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std < 1e-12:
        return 0.0
    return float(returns.mean() / std * _ANNUALIZE)


def _sanitize_signal_array(signal: np.ndarray) -> np.ndarray:
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class CombinerConfig:
    max_correlation: float = 0.3
    max_alphas: int = 30


def select_low_correlation(
    signals: np.ndarray,
    quality_scores: np.ndarray,
    config: CombinerConfig | None = None,
) -> list[int]:
    """Greedy forward selection of low-correlation alphas.

    Parameters
    ----------
    signals : (n_alphas, n_days) signal matrix
    quality_scores : (n_alphas,) quality scores for ranking
    config : combiner configuration

    Returns
    -------
    List of selected alpha indices.
    """
    cfg = config or CombinerConfig()
    n = signals.shape[0]
    if n == 0:
        return []

    priority = np.nan_to_num(np.asarray(quality_scores, dtype=np.float64), nan=0.0)
    if np.max(priority) <= 0:
        priority = compute_diversity_scores(
            _sanitize_signal_array(np.asarray(signals, dtype=np.float64))
        )
    order = np.argsort(-priority)
    selected: list[int] = [int(order[0])]

    for idx in order[1:]:
        if len(selected) >= cfg.max_alphas:
            break
        idx = int(idx)
        sig = signals[idx]
        sig_clean = _sanitize_signal_array(sig)

        # Check correlation with all selected
        too_correlated = False
        for sel_idx in selected:
            sel_sig = _sanitize_signal_array(signals[sel_idx])
            n_pts = min(len(sig_clean), len(sel_sig))
            if n_pts < 10:
                continue
            cand = sig_clean[:n_pts]
            base = sel_sig[:n_pts]
            if np.std(cand) <= 1e-12 or np.std(base) <= 1e-12:
                continue
            corr = np.corrcoef(cand, base)[0, 1]
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
    selected = _sanitize_signal_array(selected)
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
    quality_scores: np.ndarray,
    diversity: np.ndarray,
    min_weight: float = 1e-4,
) -> np.ndarray:
    """Compute normalized weights = quality × diversity + min_weight.

    Parameters
    ----------
    quality_scores : (N,) rolling quality metric values
    diversity : (N,) diversity scores in [0, 1]
    min_weight : floor so no alpha is fully excluded

    Returns
    -------
    weights : (N,) normalized weights summing to 1.0
    """
    quality = np.maximum(np.asarray(quality_scores, dtype=np.float64), 0.0)
    diversity = np.clip(np.asarray(diversity, dtype=np.float64), 0.0, 1.0)
    if np.max(quality) > 0:
        raw = quality * diversity + min_weight
    else:
        raw = diversity + min_weight
    total = raw.sum()
    if total > 0:
        return raw / total
    return np.full(len(quality_scores), 1.0 / max(len(quality_scores), 1))


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
    combined = weights @ _sanitize_signal_array(signals)
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


def signal_consensus(
    signals: dict[str, float],
    weights: dict[str, float],
) -> tuple[float, float, float]:
    """Weighted signal mean, std, and consensus from alpha signals.

    consensus = |mean| / (|mean| + std) measures how much alphas agree
    on direction.  Returns (mean, std, consensus).
    """
    if not signals:
        return 0.0, 0.0, 0.0
    ids = list(signals.keys())
    w = np.array([weights.get(a, 0.0) for a in ids])
    s = np.array([signals[a] for a in ids])
    w_sum = w.sum()
    if w_sum <= 0:
        return 0.0, 0.0, 0.0
    w_norm = w / w_sum
    mean = float(np.dot(w_norm, s))
    var = float(np.dot(w_norm, (s - mean) ** 2))
    std = sqrt(max(var, 0.0))
    abs_mean = abs(mean)
    consensus = abs_mean / (abs_mean + std) if (abs_mean + std) > 1e-12 else 0.0
    return mean, std, consensus
