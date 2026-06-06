from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from alpha_os.trading_strategy import TradingStrategy

from strategies.crypto_momentum.strategy import (
    MomentumDecisionInput,
    SevenDayMomentumStrategy,
    SevenDayMomentumWithThirtyDayTrendStrategy,
    SevenDayMomentumWithThirtyDayTrendSkfolioMaxRatioStrategy,
    TargetWeights,
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
}
