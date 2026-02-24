"""Deflated Sharpe Ratio (DSR) — Bailey & López de Prado (2014).

Adjusts a strategy's Sharpe ratio for the number of trials (selection bias),
skewness, and kurtosis of returns. Returns the probability that the observed
Sharpe is above zero after deflation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DSRResult:
    observed_sharpe: float
    expected_max_sharpe: float
    deflated_sharpe: float
    p_value: float
    is_significant: bool


def _expected_max_sharpe(n_trials: int, sharpe_std: float = 1.0) -> float:
    """E[max(SR)] under null, approximated via Euler-Mascheroni."""
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649
    z = stats.norm.ppf(1 - 1 / n_trials)
    return sharpe_std * (z - gamma / z)


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    annualization: float = 252.0,
    significance: float = 0.05,
) -> DSRResult:
    """Compute DSR for a return series given number of independent trials.

    Parameters
    ----------
    returns : array of daily returns
    n_trials : number of strategies tested (selection bias adjustment)
    annualization : trading days per year
    significance : p-value threshold for significance
    """
    n = len(returns)
    if n < 10 or n_trials < 1:
        return DSRResult(0.0, 0.0, 0.0, 1.0, False)

    sr = float(np.mean(returns) / (np.std(returns, ddof=1) + 1e-12) * np.sqrt(annualization))
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=True))

    sr_std = np.sqrt(
        (1 - skew * sr / np.sqrt(annualization) + (kurt - 1) / 4 * sr**2 / annualization) / n
    )

    expected_max = _expected_max_sharpe(n_trials, sharpe_std=1.0)

    if sr_std < 1e-12:
        return DSRResult(sr, expected_max, 0.0, 1.0, False)

    dsr_stat = (sr - expected_max) / sr_std
    p_value = 1 - stats.norm.cdf(dsr_stat)

    return DSRResult(
        observed_sharpe=sr,
        expected_max_sharpe=expected_max,
        deflated_sharpe=float(dsr_stat),
        p_value=float(p_value),
        is_significant=p_value < significance,
    )
