"""Cross-asset alpha evaluation — score one expression across multiple assets."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from alpha_os.backtest.benchmark import build_benchmark_returns
from alpha_os.backtest.cost_model import CostModel
from alpha_os.backtest.engine import BacktestEngine
from alpha_os.backtest.metrics import rank_ic, risk_adjusted_ic
from alpha_os.alpha.evaluator import sanitize_signal
from alpha_os.dsl import parse

logger = logging.getLogger(__name__)

IC_METRICS = {"ic", "ric"}
DEFAULT_HORIZONS = (1, 5, 20)


def _forward_returns(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Compute h-day forward returns: prices[t+h]/prices[t] - 1."""
    if horizon == 1:
        return np.diff(prices) / prices[:-1]
    return prices[horizon:] / prices[:-horizon] - 1


def _residualize(
    fwd_returns: np.ndarray,
    bm_returns: np.ndarray | None,
    start_idx: int,
    horizon: int,
) -> np.ndarray:
    """Subtract benchmark returns from forward returns."""
    if bm_returns is None:
        return fwd_returns
    bm_start = max(start_idx - 1, 0)
    bm_slice = bm_returns[bm_start : bm_start + len(fwd_returns)]
    if len(bm_slice) == 0:
        return fwd_returns
    n = min(len(fwd_returns), len(bm_slice))
    residual = fwd_returns[:n] - bm_slice[:n]
    return residual


@dataclass
class CrossAssetResult:
    """Result of multi-horizon cross-asset evaluation."""
    best_fitness: float = 0.0
    best_horizon: int = 1
    fitness_by_horizon: dict[int, float] = field(default_factory=dict)
    per_asset: dict[str, float] = field(default_factory=dict)


def evaluate_cross_asset(
    expression: str,
    data: dict[str, np.ndarray],
    asset_price_signals: list[str],
    *,
    fitness_metric: str = "sharpe",
    horizons: tuple[int, ...] | list[int] = (1,),
    commission_pct: float = 0.10,
    slippage_pct: float = 0.05,
    allow_short: bool = True,
    benchmark_assets: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate one alpha expression across multiple assets.

    Returns {asset_signal: fitness} for each asset where the expression
    can be evaluated. Uses the best horizon when multiple are given.
    """
    expr = parse(expression)
    use_ic = fitness_metric in IC_METRICS

    engine = None
    bm_returns = None
    if not use_ic:
        engine = BacktestEngine(
            CostModel(commission_pct, slippage_pct),
            allow_short=allow_short,
        )
    if benchmark_assets:
        bm_returns = build_benchmark_returns(data, benchmark_assets)
        if len(bm_returns) == 0:
            bm_returns = None

    if use_ic and len(horizons) > 1:
        # Multi-horizon: evaluate at each horizon, return best
        result = _evaluate_multi_horizon_ic(
            expr, data, asset_price_signals, horizons,
            fitness_metric=fitness_metric, bm_returns=bm_returns,
        )
        return result.per_asset

    results: dict[str, float] = {}
    h = horizons[0] if horizons else 1
    for price_signal in asset_price_signals:
        prices = data.get(price_signal)
        if prices is None or len(prices) < 200:
            continue
        try:
            valid = np.where(np.isfinite(prices))[0]
            if len(valid) < 200:
                continue
            start_idx, end_idx = int(valid[0]), int(valid[-1]) + 1

            sig = sanitize_signal(expr.evaluate(data))
            if sig.ndim == 0:
                sig = np.full(len(prices), float(sig))

            sig_slice = sig[start_idx:end_idx]
            prices_slice = prices[start_idx:end_idx]
            if len(sig_slice) < 200 or len(prices_slice) < 200:
                continue

            if use_ic:
                fwd = _forward_returns(prices_slice, h)
                fwd = _residualize(fwd, bm_returns, start_idx, h)
                sig_for_ic = sig_slice[: len(fwd)]
                if fitness_metric == "ric":
                    fitness = risk_adjusted_ic(sig_for_ic, fwd)
                else:
                    fitness = rank_ic(sig_for_ic, fwd)
            else:
                bm_slice = None
                if bm_returns is not None:
                    bm_start = max(start_idx - 1, 0)
                    bm_end = end_idx - 1
                    bm_slice = bm_returns[bm_start:bm_end]
                    if len(bm_slice) == 0:
                        bm_slice = None
                result = engine.run(sig_slice, prices_slice,
                                    benchmark_returns=bm_slice)
                fitness = result.fitness(fitness_metric)

            if np.isfinite(fitness) and fitness != 0.0:
                results[price_signal] = fitness
        except Exception:
            continue

    return results


def _evaluate_multi_horizon_ic(
    expr,
    data: dict[str, np.ndarray],
    asset_price_signals: list[str],
    horizons: tuple[int, ...] | list[int],
    *,
    fitness_metric: str = "ic",
    bm_returns: np.ndarray | None = None,
) -> CrossAssetResult:
    """Evaluate IC at multiple horizons, return best."""
    ic_fn = risk_adjusted_ic if fitness_metric == "ric" else rank_ic

    # per_horizon[h] = {asset: ic}
    per_horizon: dict[int, dict[str, float]] = {h: {} for h in horizons}

    for price_signal in asset_price_signals:
        prices = data.get(price_signal)
        if prices is None or len(prices) < 200:
            continue
        try:
            valid = np.where(np.isfinite(prices))[0]
            if len(valid) < 200:
                continue
            start_idx, end_idx = int(valid[0]), int(valid[-1]) + 1

            sig = sanitize_signal(expr.evaluate(data))
            if sig.ndim == 0:
                sig = np.full(len(prices), float(sig))

            sig_slice = sig[start_idx:end_idx]
            prices_slice = prices[start_idx:end_idx]
            if len(sig_slice) < 200 or len(prices_slice) < 200:
                continue

            for h in horizons:
                if len(prices_slice) < h + 200:
                    continue
                fwd = _forward_returns(prices_slice, h)
                fwd = _residualize(fwd, bm_returns, start_idx, h)
                sig_for_ic = sig_slice[: len(fwd)]
                ic = ic_fn(sig_for_ic, fwd)
                if np.isfinite(ic) and ic != 0.0:
                    per_horizon[h][price_signal] = ic
        except Exception:
            continue

    # Find best horizon by mean IC
    fitness_by_horizon: dict[int, float] = {}
    for h in horizons:
        vals = list(per_horizon[h].values())
        fitness_by_horizon[h] = float(np.mean(vals)) if vals else 0.0

    best_h = max(horizons, key=lambda h: fitness_by_horizon[h])
    best_fitness = fitness_by_horizon[best_h]

    return CrossAssetResult(
        best_fitness=best_fitness,
        best_horizon=best_h,
        fitness_by_horizon=fitness_by_horizon,
        per_asset=per_horizon[best_h],
    )


def evaluate_cross_asset_multi_horizon(
    expression: str,
    data: dict[str, np.ndarray],
    asset_price_signals: list[str],
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    fitness_metric: str = "ic",
    benchmark_assets: list[str] | None = None,
) -> CrossAssetResult:
    """Evaluate expression at multiple horizons, return full result."""
    expr = parse(expression)
    bm_returns = None
    if benchmark_assets:
        bm_returns = build_benchmark_returns(data, benchmark_assets)
        if len(bm_returns) == 0:
            bm_returns = None
    return _evaluate_multi_horizon_ic(
        expr, data, asset_price_signals, horizons,
        fitness_metric=fitness_metric, bm_returns=bm_returns,
    )


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
