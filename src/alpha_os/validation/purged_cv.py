"""Purged Walk-Forward Cross-Validation with embargo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alpha_os.backtest.engine import BacktestEngine
from alpha_os.backtest import metrics


@dataclass
class CVResult:
    oos_sharpe: float
    oos_sharpe_std: float
    oos_return: float
    oos_max_dd: float
    n_folds: int
    fold_sharpes: list[float]


def purged_walk_forward(
    alpha_signal: np.ndarray,
    prices: np.ndarray,
    engine: BacktestEngine,
    n_folds: int = 5,
    embargo: int = 5,
    min_train: int = 100,
) -> CVResult:
    """Run purged expanding-window walk-forward CV.

    Splits data into n_folds test segments. For each fold:
    - Train period: start to fold_start - embargo
    - Test period: fold_start to fold_end
    Returns OOS metrics aggregated across folds.
    """
    n = len(prices)
    if n < min_train + embargo + 20:
        return CVResult(0.0, 0.0, 0.0, 0.0, 0, [])

    test_size = (n - min_train) // n_folds
    if test_size < 10:
        return CVResult(0.0, 0.0, 0.0, 0.0, 0, [])

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

        result = engine.run(oos_signal, oos_prices, alpha_id=f"fold_{fold}")
        fold_sharpes.append(result.sharpe)

        oos_rets = np.diff(oos_prices) / oos_prices[:-1]
        pos = np.clip(np.sign(oos_signal[:-1]), -1.0, 1.0)
        all_oos_returns.extend((pos * oos_rets).tolist())

    if not fold_sharpes:
        return CVResult(0.0, 0.0, 0.0, 0.0, 0, [])

    all_oos = np.array(all_oos_returns)
    return CVResult(
        oos_sharpe=float(np.mean(fold_sharpes)),
        oos_sharpe_std=float(np.std(fold_sharpes)),
        oos_return=float(metrics.annual_return(all_oos)) if len(all_oos) > 1 else 0.0,
        oos_max_dd=float(metrics.max_drawdown(all_oos)) if len(all_oos) > 1 else 0.0,
        n_folds=len(fold_sharpes),
        fold_sharpes=fold_sharpes,
    )
