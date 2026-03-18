"""Behavior descriptor computation for MAP-Elites archive.

4 behavioral axes: persistence, activity, price_beta, vol_sensitivity.
"""
from __future__ import annotations

import numpy as np

from ..dsl.expr import Expr


def compute_behavior(
    signal: np.ndarray,
    expr: Expr,
    prices: np.ndarray | None = None,
) -> np.ndarray:
    """Compute 4D behavior descriptor for an alpha signal.

    Axes:
      0: persistence — signal autocorrelation half-life (days)
      1: activity — fraction of days with non-trivial signal
      2: price_beta — correlation between signal and recent returns
      3: vol_sensitivity — correlation between |signal| and return volatility
    """
    s = _clean(signal)
    rets = _returns(prices) if prices is not None else None
    return np.array([
        _persistence(s),
        _activity(s),
        _price_beta(s, rets),
        _vol_sensitivity(s, rets),
    ])


def _clean(signal: np.ndarray) -> np.ndarray:
    s = np.asarray(signal, dtype=float)
    mask = np.isfinite(s)
    if not mask.all():
        s = s.copy()
        s[~mask] = 0.0
    return s


def _returns(prices: np.ndarray) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    r = np.diff(p) / (np.abs(p[:-1]) + 1e-12)
    return r


def _persistence(s: np.ndarray) -> float:
    """Signal autocorrelation half-life in days."""
    if len(s) < 10 or s.std() == 0:
        return 0.0
    ac1 = np.corrcoef(s[:-1], s[1:])[0, 1]
    if np.isnan(ac1) or ac1 <= 0 or ac1 >= 1.0:
        return 0.0
    return float(-np.log(2) / np.log(ac1))


def _activity(s: np.ndarray) -> float:
    """Fraction of days with non-trivial signal magnitude."""
    if len(s) == 0:
        return 0.0
    std = s.std()
    if std == 0:
        return 1.0
    threshold = std * 0.1
    return float(np.mean(np.abs(s) > threshold))


def _price_beta(s: np.ndarray, rets: np.ndarray | None) -> float:
    """Correlation between signal and recent price returns.

    Positive = momentum/trend-following, negative = mean-reversion.
    """
    if rets is None or len(rets) < 10:
        return 0.0
    n = min(len(s), len(rets))
    sig = s[-n:]
    ret = rets[-n:]
    if sig.std() == 0 or ret.std() == 0:
        return 0.0
    c = np.corrcoef(sig, ret)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def _vol_sensitivity(s: np.ndarray, rets: np.ndarray | None) -> float:
    """Correlation between signal magnitude and return volatility.

    Positive = active in volatile markets, negative = active in calm markets.
    """
    if rets is None or len(rets) < 30:
        return 0.0
    n = min(len(s), len(rets))
    sig_mag = np.abs(s[-n:])
    # 10-day rolling volatility
    window = min(10, n // 3)
    if window < 3:
        return 0.0
    r = rets[-n:]
    # Simple rolling std using convolution
    r2 = r ** 2
    kernel = np.ones(window) / window
    rolling_var = np.convolve(r2, kernel, mode="valid")
    rolling_vol = np.sqrt(np.maximum(rolling_var, 0.0))
    # Align signal magnitude to rolling_vol length
    offset = n - len(rolling_vol)
    sig_aligned = sig_mag[offset:]
    if len(sig_aligned) < 10 or sig_aligned.std() == 0 or rolling_vol.std() == 0:
        return 0.0
    c = np.corrcoef(sig_aligned, rolling_vol)[0, 1]
    return float(c) if np.isfinite(c) else 0.0
