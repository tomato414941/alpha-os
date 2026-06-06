from __future__ import annotations

import numpy as np
from skfolio.optimization import HierarchicalRiskParity, MeanRisk, ObjectiveFunction

from strategies.crypto.momentum import MomentumDecisionInput, TargetWeights


class EqualWeightAllocator:
    def allocate(
        self,
        *,
        active_symbols: tuple[str, ...],
        strategy_input: MomentumDecisionInput,
    ) -> TargetWeights:
        if not active_symbols:
            return TargetWeights(target_weights={})
        weight = 1.0 / len(active_symbols)
        return TargetWeights(target_weights={symbol: weight for symbol in active_symbols})


class SkfolioMaxRatioAllocator:
    def allocate(
        self,
        *,
        active_symbols: tuple[str, ...],
        strategy_input: MomentumDecisionInput,
    ) -> TargetWeights:
        if not active_symbols:
            return TargetWeights(target_weights={})
        if len(active_symbols) == 1:
            return TargetWeights(target_weights={active_symbols[0]: 1.0})

        returns = _returns_matrix(
            symbols=active_symbols,
            closes_by_symbol=strategy_input.closes_by_symbol,
        )
        if returns.shape[0] < 2:
            return EqualWeightAllocator().allocate(
                active_symbols=active_symbols,
                strategy_input=strategy_input,
            )

        try:
            optimizer = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO)
            optimizer.fit(returns)
            weights = tuple(float(value) for value in optimizer.weights_)
        except Exception:
            return EqualWeightAllocator().allocate(
                active_symbols=active_symbols,
                strategy_input=strategy_input,
            )

        return TargetWeights(
            target_weights=_normalize_weights(
                dict(zip(active_symbols, weights, strict=True))
            )
        )


class SkfolioHierarchicalRiskParityAllocator:
    def allocate(
        self,
        *,
        active_symbols: tuple[str, ...],
        strategy_input: MomentumDecisionInput,
    ) -> TargetWeights:
        if not active_symbols:
            return TargetWeights(target_weights={})
        if len(active_symbols) == 1:
            return TargetWeights(target_weights={active_symbols[0]: 1.0})

        returns = _returns_matrix(
            symbols=active_symbols,
            closes_by_symbol=strategy_input.closes_by_symbol,
        )
        if returns.shape[0] < 2:
            return EqualWeightAllocator().allocate(
                active_symbols=active_symbols,
                strategy_input=strategy_input,
            )

        try:
            optimizer = HierarchicalRiskParity()
            optimizer.fit(returns)
            weights = tuple(float(value) for value in optimizer.weights_)
        except Exception:
            return EqualWeightAllocator().allocate(
                active_symbols=active_symbols,
                strategy_input=strategy_input,
            )

        return TargetWeights(
            target_weights=_normalize_weights(
                dict(zip(active_symbols, weights, strict=True))
            )
        )


class SkfolioMinimumVarianceAllocator:
    def allocate(
        self,
        *,
        active_symbols: tuple[str, ...],
        strategy_input: MomentumDecisionInput,
    ) -> TargetWeights:
        if not active_symbols:
            return TargetWeights(target_weights={})
        if len(active_symbols) == 1:
            return TargetWeights(target_weights={active_symbols[0]: 1.0})

        returns = _returns_matrix(
            symbols=active_symbols,
            closes_by_symbol=strategy_input.closes_by_symbol,
        )
        if returns.shape[0] < 2:
            return EqualWeightAllocator().allocate(
                active_symbols=active_symbols,
                strategy_input=strategy_input,
            )

        try:
            optimizer = MeanRisk()
            optimizer.fit(returns)
            weights = tuple(float(value) for value in optimizer.weights_)
        except Exception:
            return EqualWeightAllocator().allocate(
                active_symbols=active_symbols,
                strategy_input=strategy_input,
            )

        return TargetWeights(
            target_weights=_normalize_weights(
                dict(zip(active_symbols, weights, strict=True))
            )
        )


def _returns_matrix(
    *,
    symbols: tuple[str, ...],
    closes_by_symbol: dict[str, tuple[float, ...]],
) -> np.ndarray:
    columns = []
    for symbol in symbols:
        closes = closes_by_symbol[symbol]
        returns = [
            (current_close / previous_close) - 1.0
            if previous_close > 0.0
            else 0.0
            for previous_close, current_close in zip(closes[:-1], closes[1:], strict=True)
        ]
        columns.append(returns)
    return np.array(columns, dtype=float).T


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    non_negative_weights = {
        symbol: max(weight, 0.0)
        for symbol, weight in weights.items()
    }
    total_weight = sum(non_negative_weights.values())
    if total_weight <= 0.0:
        weight = 1.0 / len(non_negative_weights)
        return {symbol: weight for symbol in non_negative_weights}
    return {
        symbol: weight / total_weight
        for symbol, weight in non_negative_weights.items()
    }
