from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

_ANNUALIZE = sqrt(252)


def compute_tc_scores(
    signal_arrays: dict[str, np.ndarray],
    returns: np.ndarray,
    min_observations: int = 20,
) -> dict[str, float]:
    ids = list(signal_arrays.keys())
    n = len(ids)
    if n == 0:
        return {}

    finite_mask = np.isfinite(returns)
    clean_returns = returns[finite_mask]
    if len(clean_returns) < min_observations:
        return {hypothesis_id: 0.0 for hypothesis_id in ids}

    signals = np.array(
        [
            _sanitize_signal_array(signal_arrays[hypothesis_id][-len(returns):])[finite_mask]
            for hypothesis_id in ids
        ]
    )

    ensemble = signals.mean(axis=0) * clean_returns
    full_sharpe = _sharpe(ensemble)

    scores: dict[str, float] = {}
    for idx, hypothesis_id in enumerate(ids):
        if n == 1:
            scores[hypothesis_id] = full_sharpe
            continue
        alpha_mask = np.ones(n, dtype=bool)
        alpha_mask[idx] = False
        ensemble_without = signals[alpha_mask].mean(axis=0) * clean_returns
        scores[hypothesis_id] = full_sharpe - _sharpe(ensemble_without)
    return scores


def compute_tc_weights(
    tc_scores: dict[str, float],
    min_weight: float = 1e-4,
    max_weight: float = 0.3,
) -> dict[str, float]:
    if not tc_scores:
        return {}
    ids = list(tc_scores.keys())
    raw = np.array([max(tc_scores[hypothesis_id], 0.0) + min_weight for hypothesis_id in ids])
    raw = np.minimum(raw, max_weight * raw.sum())
    total = raw.sum()
    if total <= 0:
        eq = 1.0 / len(ids)
        return {hypothesis_id: eq for hypothesis_id in ids}
    weights = raw / total
    return {hypothesis_id: float(weights[idx]) for idx, hypothesis_id in enumerate(ids)}


def compute_stake_weights(
    stakes: dict[str, float],
    max_weight: float = 0.05,
) -> dict[str, float]:
    if not stakes:
        return {}
    ids = list(stakes.keys())
    raw = np.array([max(stakes[hypothesis_id], 0.0) for hypothesis_id in ids])
    total = raw.sum()
    if total <= 0:
        eq = 1.0 / len(ids) if ids else 0.0
        return {hypothesis_id: eq for hypothesis_id in ids}
    weights = raw / total
    weights = np.minimum(weights, max_weight)
    weight_total = weights.sum()
    if weight_total > 0:
        weights = weights / weight_total
    return {hypothesis_id: float(weights[idx]) for idx, hypothesis_id in enumerate(ids)}


@dataclass
class CombinerConfig:
    max_correlation: float = 0.3
    max_alphas: int = 30


def select_low_correlation(
    signals: np.ndarray,
    quality_scores: np.ndarray,
    config: CombinerConfig | None = None,
) -> list[int]:
    cfg = config or CombinerConfig()
    n = signals.shape[0]
    if n == 0:
        return []

    priority = np.nan_to_num(np.asarray(quality_scores, dtype=np.float64), nan=0.0)
    order = np.argsort(-priority)
    selected: list[int] = [int(order[0])]

    for idx in order[1:]:
        if len(selected) >= cfg.max_alphas:
            break
        idx = int(idx)
        sig = signals[idx]
        sig_clean = _sanitize_signal_array(sig)

        too_correlated = False
        for selected_idx in selected:
            selected_signal = _sanitize_signal_array(signals[selected_idx])
            n_points = min(len(sig_clean), len(selected_signal))
            if n_points < 10:
                continue
            candidate = sig_clean[:n_points]
            base = selected_signal[:n_points]
            if np.std(candidate) <= 1e-12 or np.std(base) <= 1e-12:
                continue
            corr = np.corrcoef(candidate, base)[0, 1]
            if np.isnan(corr):
                continue
            if corr > cfg.max_correlation:
                too_correlated = True
                break

        if not too_correlated:
            selected.append(idx)

    return selected


def cross_asset_neutralize(
    signals: dict[str, float],
) -> dict[str, float]:
    if len(signals) <= 1:
        return dict(signals)
    values = np.array(list(signals.values()))
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return {key: 0.0 for key in signals}
    mean = float(np.mean(values[finite_mask]))
    return {
        key: (float(value - mean) if np.isfinite(value) else 0.0)
        for key, value in signals.items()
    }


def weighted_combine_scalar(
    signals: dict[str, float],
    weights: dict[str, float],
) -> float:
    total = 0.0
    weight_sum = 0.0
    for hypothesis_id, signal in signals.items():
        if not np.isfinite(signal):
            continue
        weight = weights.get(hypothesis_id, 0.0)
        total += weight * signal
        weight_sum += weight
    if weight_sum > 0 and weight_sum < 0.99:
        total /= weight_sum
    return float(np.clip(total, -1.0, 1.0))


def signal_consensus(
    signals: dict[str, float],
    weights: dict[str, float],
) -> tuple[float, float, float]:
    if not signals:
        return 0.0, 0.0, 0.0
    ids = list(signals.keys())
    weight_values = np.array([weights.get(hypothesis_id, 0.0) for hypothesis_id in ids])
    signal_values = np.array([signals[hypothesis_id] for hypothesis_id in ids])
    finite_mask = np.isfinite(signal_values)
    if not finite_mask.all():
        weight_values = weight_values[finite_mask]
        signal_values = signal_values[finite_mask]
    if len(signal_values) == 0:
        return 0.0, 0.0, 0.0
    weight_sum = weight_values.sum()
    if weight_sum <= 0:
        return 0.0, 0.0, 0.0
    normalized = weight_values / weight_sum
    mean = float(np.dot(normalized, signal_values))
    variance = float(np.dot(normalized, (signal_values - mean) ** 2))
    std = sqrt(max(variance, 0.0))
    abs_mean = abs(mean)
    consensus = abs_mean / (abs_mean + std) if (abs_mean + std) > 1e-12 else 0.0
    return mean, std, consensus


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std < 1e-12:
        return 0.0
    return float(returns.mean() / std * _ANNUALIZE)


def _sanitize_signal_array(signal: np.ndarray) -> np.ndarray:
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
