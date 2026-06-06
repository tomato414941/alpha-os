from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from alpha_os.trading_strategy import TradingStrategy

from strategies.crypto.momentum import (
    MomentumDecisionInput,
    SevenDayMomentumStrategy,
    SevenDayMomentumWithThirtyDayTrendSkfolioHrpStrategy,
    SevenDayMomentumWithThirtyDayTrendSkfolioMaxRatioStrategy,
    SevenDayMomentumWithThirtyDayTrendStrategy,
    TargetWeights,
)
from strategies.crypto.pullback import (
    ThreeDayPullbackWithThirtyDayTrendSkfolioMaxRatioStrategy,
    ThreeDayPullbackWithThirtyDayTrendStrategy,
)


CURRENT_VARIANT = "7d_momentum_30d_trend"


@dataclass(frozen=True)
class StrategyVariant:
    factory: Callable[[], TradingStrategy[MomentumDecisionInput, TargetWeights]]
    lookback_days: int


VARIANTS = {
    "7d_momentum": StrategyVariant(
        factory=SevenDayMomentumStrategy,
        lookback_days=7,
    ),
    "7d_momentum_30d_trend": StrategyVariant(
        factory=SevenDayMomentumWithThirtyDayTrendStrategy,
        lookback_days=30,
    ),
    "7d_momentum_30d_trend_skfolio_max_ratio": StrategyVariant(
        factory=SevenDayMomentumWithThirtyDayTrendSkfolioMaxRatioStrategy,
        lookback_days=30,
    ),
    "7d_momentum_30d_trend_skfolio_hrp": StrategyVariant(
        factory=SevenDayMomentumWithThirtyDayTrendSkfolioHrpStrategy,
        lookback_days=30,
    ),
    "3d_pullback_30d_trend": StrategyVariant(
        factory=ThreeDayPullbackWithThirtyDayTrendStrategy,
        lookback_days=30,
    ),
    "3d_pullback_30d_trend_skfolio_max_ratio": StrategyVariant(
        factory=ThreeDayPullbackWithThirtyDayTrendSkfolioMaxRatioStrategy,
        lookback_days=30,
    ),
}
