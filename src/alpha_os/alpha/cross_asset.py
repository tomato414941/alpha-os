"""Cross-asset alpha evaluation — score one expression across multiple assets."""
from __future__ import annotations

import logging

import numpy as np

from alpha_os.backtest.benchmark import build_benchmark_returns
from alpha_os.backtest.cost_model import CostModel
from alpha_os.backtest.engine import BacktestEngine
from alpha_os.alpha.evaluator import sanitize_signal
from alpha_os.dsl import parse

logger = logging.getLogger(__name__)


def evaluate_cross_asset(
    expression: str,
    data: dict[str, np.ndarray],
    asset_price_signals: list[str],
    *,
    fitness_metric: str = "sharpe",
    commission_pct: float = 0.10,
    slippage_pct: float = 0.05,
    allow_short: bool = True,
    benchmark_assets: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate one alpha expression across multiple assets.

    Returns {asset_signal: fitness} for each asset where the expression
    can be evaluated. Assets with missing data or failed evaluation are
    excluded.

    Parameters
    ----------
    expression : DSL expression string
    data : {signal_name: values_array} full data dict
    asset_price_signals : list of price signal names to evaluate against
    fitness_metric : which fitness metric to use
    benchmark_assets : if set, compute residual fitness vs this benchmark
    """
    expr = parse(expression)
    engine = BacktestEngine(
        CostModel(commission_pct, slippage_pct),
        allow_short=allow_short,
    )

    bm_returns = None
    if benchmark_assets:
        bm_returns = build_benchmark_returns(data, benchmark_assets)
        if len(bm_returns) == 0:
            bm_returns = None

    results: dict[str, float] = {}
    for price_signal in asset_price_signals:
        prices = data.get(price_signal)
        if prices is None or len(prices) < 200:
            continue
        try:
            sig = sanitize_signal(expr.evaluate(data))
            if sig.ndim == 0:
                sig = np.full(len(prices), float(sig))
            n = min(len(sig), len(prices))
            sig = sig[:n]
            prices_trimmed = prices[:n]
            result = engine.run(sig, prices_trimmed,
                                benchmark_returns=bm_returns)
            fitness = result.fitness(fitness_metric)
            if np.isfinite(fitness):
                results[price_signal] = fitness
        except Exception:
            continue

    return results


def mean_cross_asset_fitness(
    expression: str,
    data: dict[str, np.ndarray],
    asset_price_signals: list[str],
    **kwargs,
) -> float:
    """Average fitness of an expression across multiple assets.

    Returns 0.0 if no asset produced a valid fitness.
    """
    per_asset = evaluate_cross_asset(
        expression, data, asset_price_signals, **kwargs,
    )
    if not per_asset:
        return 0.0
    return float(np.mean(list(per_asset.values())))
