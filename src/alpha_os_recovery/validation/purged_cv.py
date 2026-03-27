"""Purged Walk-Forward Cross-Validation with embargo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alpha_os_recovery.backtest.engine import BacktestEngine
from alpha_os_recovery.backtest import metrics


@dataclass
class CVResult:
    oos_sharpe: float
    oos_sharpe_std: float
    oos_return: float
    oos_max_dd: float
    oos_cvar_95: float
    oos_expected_log_growth: float
    oos_tail_hit_rate: float
    n_folds: int
    fold_sharpes: list[float]

    _OOS_FITNESS_MAP = {"sharpe": "oos_sharpe", "log_growth": "oos_expected_log_growth"}

    def oos_fitness(self, metric: str = "sharpe") -> float:
        return getattr(self, self._OOS_FITNESS_MAP[metric])


def purged_walk_forward(
    alpha_signal: np.ndarray,
    prices: np.ndarray,
    engine: BacktestEngine,
    n_folds: int = 5,
    embargo: int = 5,
    min_train: int = 100,
    benchmark_returns: np.ndarray | None = None,
) -> CVResult:
    """Run purged expanding-window walk-forward CV.

    Splits data into n_folds test segments. For each fold:
    - Train period: start to fold_start - embargo
    - Test period: fold_start to fold_end
    Returns OOS metrics aggregated across folds.
    """
    n = len(prices)
    if n < min_train + embargo + 20:
        return CVResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, [])

    test_size = (n - min_train) // n_folds
    if test_size < 10:
        return CVResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, [])

    fold_sharpes = []
    all_oos_returns = []

    for fold in range(n_folds):
        test_start = min_train + fold * test_size
        test_end = min(test_start + test_size, n)
        if test_end - test_start < 10:
            continue

        # OOS evaluation: positions from alpha, returns from prices
        oos_signal = alpha_signal[test_start:test_end]
        oos_prices = prices[test_start:test_end]

        if len(oos_prices) < 10:
            continue

        oos_bm = None
        if benchmark_returns is not None:
            oos_bm = benchmark_returns[test_start:test_end - 1]
        result = engine.run(oos_signal, oos_prices,
                            alpha_id=f"fold_{fold}", benchmark_returns=oos_bm)
        fold_sharpes.append(result.sharpe)

        oos_rets = np.diff(oos_prices) / oos_prices[:-1]
        pos = engine.positions(oos_signal)[:-1]
        fold_net = pos * oos_rets
        if oos_bm is not None:
            n_bm = min(len(fold_net), len(oos_bm))
            fold_net[:n_bm] = fold_net[:n_bm] - oos_bm[:n_bm]
        all_oos_returns.extend(fold_net.tolist())

    if not fold_sharpes:
        return CVResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, [])

    all_oos = np.array(all_oos_returns)
    return CVResult(
        oos_sharpe=float(np.mean(fold_sharpes)),
        oos_sharpe_std=float(np.std(fold_sharpes)),
        oos_return=float(metrics.annual_return(all_oos)) if len(all_oos) > 1 else 0.0,
        oos_max_dd=float(metrics.max_drawdown(all_oos)) if len(all_oos) > 1 else 0.0,
        oos_cvar_95=float(metrics.cvar(all_oos, alpha=0.05)) if len(all_oos) > 1 else 0.0,
        oos_expected_log_growth=float(metrics.expected_log_growth(all_oos))
        if len(all_oos) > 1 else 0.0,
        oos_tail_hit_rate=float(metrics.tail_hit_rate(all_oos, sigma=2.0))
        if len(all_oos) > 1 else 0.0,
        n_folds=len(fold_sharpes),
        fold_sharpes=fold_sharpes,
    )


@dataclass
class ICCVResult:
    """Result of IC-based walk-forward cross-validation."""
    oos_ic: float
    oos_ic_std: float
    n_folds: int
    fold_ics: list[float]
    # Keep oos_sharpe for backward compat with adoption gates
    oos_sharpe: float = 0.0
    oos_expected_log_growth: float = 0.0
    oos_cvar_95: float = 0.0
    oos_tail_hit_rate: float = 0.0


def purged_walk_forward_ic(
    alpha_signal: np.ndarray,
    prices: np.ndarray,
    *,
    horizon: int = 1,
    n_folds: int = 5,
    embargo: int = 5,
    min_train: int = 100,
    benchmark_returns: np.ndarray | None = None,
) -> ICCVResult:
    """Run purged walk-forward CV using IC (rank correlation) per fold.

    For each fold, computes IC between the signal and residualized
    forward returns at the given horizon. No backtest engine needed.
    """
    from alpha_os_recovery.research.cross_asset import (
        forward_returns,
        residualize_forward_returns,
    )

    n = len(prices)
    if n < min_train + embargo + horizon + 20:
        return ICCVResult(0.0, 0.0, 0, [])

    test_size = (n - min_train) // n_folds
    if test_size < max(20, horizon + 10):
        return ICCVResult(0.0, 0.0, 0, [])

    fold_ics: list[float] = []

    for fold_idx in range(n_folds):
        test_start = min_train + fold_idx * test_size
        test_end = min(test_start + test_size, n)
        if test_end - test_start < max(20, horizon + 10):
            continue

        oos_signal = alpha_signal[test_start:test_end]
        oos_prices = prices[test_start:test_end]

        fwd = forward_returns(oos_prices, horizon)
        fwd = residualize_forward_returns(fwd, benchmark_returns, test_start, horizon)
        sig_for_ic = oos_signal[: len(fwd)]

        ic = metrics.rank_ic(sig_for_ic, fwd)
        fold_ics.append(ic)

    if not fold_ics:
        return ICCVResult(0.0, 0.0, 0, [])

    return ICCVResult(
        oos_ic=float(np.mean(fold_ics)),
        oos_ic_std=float(np.std(fold_ics)) if len(fold_ics) > 1 else 0.0,
        n_folds=len(fold_ics),
        fold_ics=fold_ics,
        oos_sharpe=float(np.mean(fold_ics)),  # approximate for gate compat
    )
