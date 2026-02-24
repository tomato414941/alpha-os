"""Behavior descriptor computation for MAP-Elites archive."""
from __future__ import annotations

import numpy as np

from ..dsl.expr import Expr
from ..dsl.generator import _collect_nodes


def compute_behavior(
    signal: np.ndarray,
    expr: Expr,
    live_signals: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Compute 4D behavior descriptor for an alpha signal.

    Axes:
      0: corr_to_live_book — avg |corr| with existing alphas (0–1)
      1: holding_half_life — signal autocorrelation half-life (days)
      2: turnover — mean daily |position change| after normalization
      3: complexity — expression tree node count
    """
    return np.array([
        _avg_abs_corr(signal, live_signals),
        _holding_half_life(signal),
        _signal_turnover(signal),
        float(len(_collect_nodes(expr))),
    ])


def _avg_abs_corr(
    signal: np.ndarray, live_signals: list[np.ndarray] | None
) -> float:
    if not live_signals:
        return 0.0
    corrs: list[float] = []
    for ls in live_signals:
        n = min(len(signal), len(ls))
        mask = ~(np.isnan(signal[:n]) | np.isnan(ls[:n]))
        if mask.sum() < 10:
            continue
        c = np.corrcoef(signal[:n][mask], ls[:n][mask])[0, 1]
        if not np.isnan(c):
            corrs.append(abs(c))
    return float(np.mean(corrs)) if corrs else 0.0


def _holding_half_life(signal: np.ndarray) -> float:
    s = signal[~np.isnan(signal)]
    if len(s) < 10:
        return 0.0
    if s.std() == 0:
        return 0.0
    ac1 = np.corrcoef(s[:-1], s[1:])[0, 1]
    if np.isnan(ac1) or ac1 <= 0:
        return 0.0
    # Half-life from AR(1): t_half = -ln(2) / ln(autocorr_lag1)
    return float(-np.log(2) / np.log(ac1))


def _signal_turnover(signal: np.ndarray) -> float:
    s = signal.copy()
    s[np.isnan(s)] = 0.0
    std = s.std()
    if std > 0:
        s = s / std
    s = np.clip(s, -1, 1)
    if len(s) < 2:
        return 0.0
    return float(np.abs(np.diff(s)).mean())
