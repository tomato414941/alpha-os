from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MomentumDecisionInput:
    closes_by_symbol: dict[str, tuple[float, ...]]
    current_weights: dict[str, float]
    equity: float


@dataclass(frozen=True)
class TargetWeights:
    target_weights: dict[str, float]


class PortfolioAllocator(Protocol):
    def allocate(
        self,
        *,
        active_symbols: tuple[str, ...],
        strategy_input: MomentumDecisionInput,
    ) -> TargetWeights:
        ...


class SevenDayMomentumStrategy:
    def decide(self, strategy_input: MomentumDecisionInput) -> TargetWeights:
        active_symbols = []
        for symbol, closes in strategy_input.closes_by_symbol.items():
            if len(closes) < 8:
                continue
            previous_close = closes[-8]
            current_close = closes[-1]
            if previous_close <= 0.0:
                continue
            if (current_close / previous_close) - 1.0 > 0.0:
                active_symbols.append(symbol)

        if not active_symbols:
            return TargetWeights(target_weights={})

        weight = 1.0 / len(active_symbols)
        return TargetWeights(
            target_weights={symbol: weight for symbol in active_symbols}
        )


class SevenDayMomentumWithThirtyDayTrendStrategy:
    def __init__(self) -> None:
        from strategies.crypto.allocation import EqualWeightAllocator

        self._strategy = TrendFilteredMomentumStrategy(
            momentum_lookback_days=7,
            trend_lookback_days=30,
            allocator=EqualWeightAllocator(),
        )

    def decide(self, strategy_input: MomentumDecisionInput) -> TargetWeights:
        return self._strategy.decide(strategy_input)


class TrendFilteredMomentumStrategy:
    def __init__(
        self,
        *,
        momentum_lookback_days: int,
        trend_lookback_days: int,
        allocator: PortfolioAllocator,
    ) -> None:
        self._momentum_lookback_days = momentum_lookback_days
        self._trend_lookback_days = trend_lookback_days
        self._allocator = allocator

    def decide(self, strategy_input: MomentumDecisionInput) -> TargetWeights:
        required_closes = max(
            self._momentum_lookback_days,
            self._trend_lookback_days,
        ) + 1
        active_symbols: list[str] = []
        for symbol, closes in strategy_input.closes_by_symbol.items():
            if len(closes) < required_closes:
                continue
            current_close = closes[-1]
            previous_momentum_close = closes[-self._momentum_lookback_days - 1]
            previous_trend_close = closes[-self._trend_lookback_days - 1]
            if previous_momentum_close <= 0.0 or previous_trend_close <= 0.0:
                continue
            momentum_return = (current_close / previous_momentum_close) - 1.0
            trend_return = (current_close / previous_trend_close) - 1.0
            if momentum_return > 0.0 and trend_return > 0.0:
                active_symbols.append(symbol)

        return self._allocator.allocate(
            active_symbols=tuple(active_symbols),
            strategy_input=strategy_input,
        )


class SevenDayMomentumWithThirtyDayTrendSkfolioMaxRatioStrategy:
    def __init__(self) -> None:
        from strategies.crypto.allocation import SkfolioMaxRatioAllocator

        self._strategy = TrendFilteredMomentumStrategy(
            momentum_lookback_days=7,
            trend_lookback_days=30,
            allocator=SkfolioMaxRatioAllocator(),
        )

    def decide(self, strategy_input: MomentumDecisionInput) -> TargetWeights:
        return self._strategy.decide(strategy_input)
