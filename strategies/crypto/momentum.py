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


class RollingEligibleTrendFilteredMomentumStrategy:
    def __init__(
        self,
        *,
        momentum_lookback_days: int,
        trend_lookback_days: int,
        eligibility_lookback_days: int,
        min_eligibility_return: float,
        max_eligibility_drawdown: float,
        allocator: PortfolioAllocator,
    ) -> None:
        self._eligibility_lookback_days = eligibility_lookback_days
        self._min_eligibility_return = min_eligibility_return
        self._max_eligibility_drawdown = max_eligibility_drawdown
        self._strategy = TrendFilteredMomentumStrategy(
            momentum_lookback_days=momentum_lookback_days,
            trend_lookback_days=trend_lookback_days,
            allocator=allocator,
        )

    def decide(self, strategy_input: MomentumDecisionInput) -> TargetWeights:
        return self._strategy.decide(
            MomentumDecisionInput(
                closes_by_symbol={
                    symbol: closes
                    for symbol, closes in strategy_input.closes_by_symbol.items()
                    if _passes_rolling_asset_quality(
                        closes,
                        lookback_days=self._eligibility_lookback_days,
                        min_return=self._min_eligibility_return,
                        max_drawdown=self._max_eligibility_drawdown,
                    )
                },
                current_weights=strategy_input.current_weights,
                equity=strategy_input.equity,
            )
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


class SevenDayMomentumWithThirtyDayTrendSkfolioMaxRatioEligibleStrategy:
    def __init__(self) -> None:
        from strategies.crypto.allocation import SkfolioMaxRatioAllocator

        self._strategy = RollingEligibleTrendFilteredMomentumStrategy(
            momentum_lookback_days=7,
            trend_lookback_days=30,
            eligibility_lookback_days=180,
            min_eligibility_return=0.0,
            max_eligibility_drawdown=-0.80,
            allocator=SkfolioMaxRatioAllocator(),
        )

    def decide(self, strategy_input: MomentumDecisionInput) -> TargetWeights:
        return self._strategy.decide(strategy_input)


class SevenDayMomentumWithThirtyDayTrendSkfolioHrpStrategy:
    def __init__(self) -> None:
        from strategies.crypto.allocation import SkfolioHierarchicalRiskParityAllocator

        self._strategy = TrendFilteredMomentumStrategy(
            momentum_lookback_days=7,
            trend_lookback_days=30,
            allocator=SkfolioHierarchicalRiskParityAllocator(),
        )

    def decide(self, strategy_input: MomentumDecisionInput) -> TargetWeights:
        return self._strategy.decide(strategy_input)


def _passes_rolling_asset_quality(
    closes: tuple[float, ...],
    *,
    lookback_days: int,
    min_return: float,
    max_drawdown: float,
) -> bool:
    if len(closes) < lookback_days + 1:
        return False
    window = closes[-lookback_days - 1 :]
    first_close = window[0]
    last_close = window[-1]
    if first_close <= 0.0:
        return False
    total_return = (last_close / first_close) - 1.0
    if total_return < min_return:
        return False

    peak = 0.0
    worst_drawdown = 0.0
    for close in window:
        peak = max(peak, close)
        if peak > 0.0:
            worst_drawdown = min(worst_drawdown, (close / peak) - 1.0)
    return worst_drawdown >= max_drawdown


class SevenDayMomentumWithThirtyDayTrendSkfolioMinimumVarianceStrategy:
    def __init__(self) -> None:
        from strategies.crypto.allocation import SkfolioMinimumVarianceAllocator

        self._strategy = TrendFilteredMomentumStrategy(
            momentum_lookback_days=7,
            trend_lookback_days=30,
            allocator=SkfolioMinimumVarianceAllocator(),
        )

    def decide(self, strategy_input: MomentumDecisionInput) -> TargetWeights:
        return self._strategy.decide(strategy_input)


class SevenDayMomentumWithThirtyDayTrendSkfolioRiskBudgetingStrategy:
    def __init__(self) -> None:
        from strategies.crypto.allocation import SkfolioRiskBudgetingAllocator

        self._strategy = TrendFilteredMomentumStrategy(
            momentum_lookback_days=7,
            trend_lookback_days=30,
            allocator=SkfolioRiskBudgetingAllocator(),
        )

    def decide(self, strategy_input: MomentumDecisionInput) -> TargetWeights:
        return self._strategy.decide(strategy_input)
