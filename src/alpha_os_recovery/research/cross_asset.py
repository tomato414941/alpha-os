from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alpha_os_recovery.backtest.benchmark import build_benchmark_returns
from alpha_os_recovery.backtest.cost_model import CostModel
from alpha_os_recovery.backtest.engine import BacktestEngine
from alpha_os_recovery.backtest.metrics import rank_ic, risk_adjusted_ic
from alpha_os_recovery.dsl import parse
from alpha_os_recovery.dsl.evaluator import sanitize_signal

IC_METRICS = {"ic", "ric"}
DEFAULT_HORIZONS = (1, 5, 20)


def forward_returns(prices: np.ndarray, horizon: int) -> np.ndarray:
    if horizon == 1:
        return np.diff(prices) / prices[:-1]
    return prices[horizon:] / prices[:-horizon] - 1


def residualize_forward_returns(
    fwd_returns: np.ndarray,
    benchmark_returns: np.ndarray | None,
    start_idx: int,
    horizon: int,
) -> np.ndarray:
    if benchmark_returns is None:
        return fwd_returns
    benchmark_start = max(start_idx - 1, 0)
    benchmark_slice = benchmark_returns[benchmark_start : benchmark_start + len(fwd_returns)]
    if len(benchmark_slice) == 0:
        return fwd_returns
    n = min(len(fwd_returns), len(benchmark_slice))
    return fwd_returns[:n] - benchmark_slice[:n]


@dataclass
class CrossAssetResult:
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
    expr = parse(expression)
    use_ic = fitness_metric in IC_METRICS

    engine = None
    benchmark_returns = None
    if not use_ic:
        engine = BacktestEngine(
            CostModel(commission_pct, slippage_pct),
            allow_short=allow_short,
        )
    if benchmark_assets:
        benchmark_returns = build_benchmark_returns(data, benchmark_assets)
        if len(benchmark_returns) == 0:
            benchmark_returns = None

    if use_ic and len(horizons) > 1:
        result = _evaluate_multi_horizon_ic(
            expr,
            data,
            asset_price_signals,
            horizons,
            fitness_metric=fitness_metric,
            benchmark_returns=benchmark_returns,
        )
        return result.per_asset

    results: dict[str, float] = {}
    horizon = horizons[0] if horizons else 1
    for price_signal in asset_price_signals:
        prices = data.get(price_signal)
        if prices is None or len(prices) < 200:
            continue
        try:
            valid = np.where(np.isfinite(prices))[0]
            if len(valid) < 200:
                continue
            start_idx, end_idx = int(valid[0]), int(valid[-1]) + 1

            signal = sanitize_signal(expr.evaluate(data))
            if signal.ndim == 0:
                signal = np.full(len(prices), float(signal))

            signal_slice = signal[start_idx:end_idx]
            prices_slice = prices[start_idx:end_idx]
            if len(signal_slice) < 200 or len(prices_slice) < 200:
                continue

            if use_ic:
                fwd = forward_returns(prices_slice, horizon)
                fwd = residualize_forward_returns(fwd, benchmark_returns, start_idx, horizon)
                signal_for_ic = signal_slice[: len(fwd)]
                if fitness_metric == "ric":
                    fitness = risk_adjusted_ic(signal_for_ic, fwd)
                else:
                    fitness = rank_ic(signal_for_ic, fwd)
            else:
                benchmark_slice = None
                if benchmark_returns is not None:
                    benchmark_start = max(start_idx - 1, 0)
                    benchmark_end = end_idx - 1
                    benchmark_slice = benchmark_returns[benchmark_start:benchmark_end]
                    if len(benchmark_slice) == 0:
                        benchmark_slice = None
                result = engine.run(signal_slice, prices_slice, benchmark_returns=benchmark_slice)
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
    benchmark_returns: np.ndarray | None = None,
) -> CrossAssetResult:
    ic_fn = risk_adjusted_ic if fitness_metric == "ric" else rank_ic
    per_horizon: dict[int, dict[str, float]] = {horizon: {} for horizon in horizons}

    for price_signal in asset_price_signals:
        prices = data.get(price_signal)
        if prices is None or len(prices) < 200:
            continue
        try:
            valid = np.where(np.isfinite(prices))[0]
            if len(valid) < 200:
                continue
            start_idx, end_idx = int(valid[0]), int(valid[-1]) + 1

            signal = sanitize_signal(expr.evaluate(data))
            if signal.ndim == 0:
                signal = np.full(len(prices), float(signal))

            signal_slice = signal[start_idx:end_idx]
            prices_slice = prices[start_idx:end_idx]
            if len(signal_slice) < 200 or len(prices_slice) < 200:
                continue

            for horizon in horizons:
                if len(prices_slice) < horizon + 200:
                    continue
                fwd = forward_returns(prices_slice, horizon)
                fwd = residualize_forward_returns(
                    fwd,
                    benchmark_returns,
                    start_idx,
                    horizon,
                )
                signal_for_ic = signal_slice[: len(fwd)]
                ic = ic_fn(signal_for_ic, fwd)
                if np.isfinite(ic) and ic != 0.0:
                    per_horizon[horizon][price_signal] = ic
        except Exception:
            continue

    fitness_by_horizon: dict[int, float] = {}
    for horizon in horizons:
        values = list(per_horizon[horizon].values())
        fitness_by_horizon[horizon] = float(np.mean(values)) if values else 0.0

    best_horizon = max(horizons, key=lambda horizon: fitness_by_horizon[horizon])
    return CrossAssetResult(
        best_fitness=fitness_by_horizon[best_horizon],
        best_horizon=best_horizon,
        fitness_by_horizon=fitness_by_horizon,
        per_asset=per_horizon[best_horizon],
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
    expr = parse(expression)
    benchmark_returns = None
    if benchmark_assets:
        benchmark_returns = build_benchmark_returns(data, benchmark_assets)
        if len(benchmark_returns) == 0:
            benchmark_returns = None
    return _evaluate_multi_horizon_ic(
        expr,
        data,
        asset_price_signals,
        horizons,
        fitness_metric=fitness_metric,
        benchmark_returns=benchmark_returns,
    )


def mean_cross_asset_fitness(
    expression: str,
    data: dict[str, np.ndarray],
    asset_price_signals: list[str],
    **kwargs,
) -> float:
    per_asset = evaluate_cross_asset(
        expression,
        data,
        asset_price_signals,
        **kwargs,
    )
    if not per_asset:
        return 0.0
    return float(np.mean(list(per_asset.values())))
