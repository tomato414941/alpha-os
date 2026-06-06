from __future__ import annotations

from strategies.crypto.momentum import (
    MomentumDecisionInput,
    PortfolioAllocator,
    TargetWeights,
)


class TrendPullbackStrategy:
    def __init__(
        self,
        *,
        pullback_lookback_days: int,
        trend_lookback_days: int,
        allocator: PortfolioAllocator,
    ) -> None:
        self._pullback_lookback_days = pullback_lookback_days
        self._trend_lookback_days = trend_lookback_days
        self._allocator = allocator

    def decide(self, strategy_input: MomentumDecisionInput) -> TargetWeights:
        required_closes = max(
            self._pullback_lookback_days,
            self._trend_lookback_days,
        ) + 1
        active_symbols: list[str] = []
        for symbol, closes in strategy_input.closes_by_symbol.items():
            if len(closes) < required_closes:
                continue
            current_close = closes[-1]
            previous_pullback_close = closes[-self._pullback_lookback_days - 1]
            previous_trend_close = closes[-self._trend_lookback_days - 1]
            if previous_pullback_close <= 0.0 or previous_trend_close <= 0.0:
                continue
            pullback_return = (current_close / previous_pullback_close) - 1.0
            trend_return = (current_close / previous_trend_close) - 1.0
            if trend_return > 0.0 and pullback_return < 0.0:
                active_symbols.append(symbol)

        return self._allocator.allocate(
            active_symbols=tuple(active_symbols),
            strategy_input=strategy_input,
        )


class ThreeDayPullbackWithThirtyDayTrendStrategy:
    def __init__(self) -> None:
        from strategies.crypto.allocation import EqualWeightAllocator

        self._strategy = TrendPullbackStrategy(
            pullback_lookback_days=3,
            trend_lookback_days=30,
            allocator=EqualWeightAllocator(),
        )

    def decide(self, strategy_input: MomentumDecisionInput) -> TargetWeights:
        return self._strategy.decide(strategy_input)


class ThreeDayPullbackWithThirtyDayTrendSkfolioMaxRatioStrategy:
    def __init__(self) -> None:
        from strategies.crypto.allocation import SkfolioMaxRatioAllocator

        self._strategy = TrendPullbackStrategy(
            pullback_lookback_days=3,
            trend_lookback_days=30,
            allocator=SkfolioMaxRatioAllocator(),
        )

    def decide(self, strategy_input: MomentumDecisionInput) -> TargetWeights:
        return self._strategy.decide(strategy_input)
