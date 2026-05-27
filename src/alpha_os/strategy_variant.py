from __future__ import annotations

from dataclasses import dataclass

from .evaluation_cost_config import TradingEnvironment
from .portfolio_construction_config import PortfolioConstructionSpec
from .trading_strategy import TradingStrategySpec


@dataclass(frozen=True)
class StrategyVariantConfig:
    portfolio_construction: PortfolioConstructionSpec
    trading_environment: TradingEnvironment
    top_k: int | None = None

    @property
    def sizing_method(self) -> str:
        return self.portfolio_construction.sizing_method

    @property
    def sizing_engine(self) -> str:
        return self.portfolio_construction.sizing_engine


def strategy_variant_config_from_strategy(
    trading_strategy: TradingStrategySpec,
) -> StrategyVariantConfig:
    return StrategyVariantConfig(
        portfolio_construction=trading_strategy.portfolio_construction,
        trading_environment=trading_strategy.trading_environment,
        top_k=trading_strategy.top_k,
    )
