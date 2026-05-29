from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from .portfolio_decision import PortfolioDecisionInput, PortfolioDecisionOutput
from .portfolio_sizing_policy import PortfolioSizingPolicy, apply_portfolio_sizing_policy


StrategyInputT = TypeVar("StrategyInputT", contravariant=True)
StrategyOutputT = TypeVar("StrategyOutputT", covariant=True)


@dataclass(frozen=True)
class TradingStrategyInput:
    pass


@dataclass(frozen=True)
class TradingStrategyOutput:
    pass


class TradingStrategy(Protocol[StrategyInputT, StrategyOutputT]):
    """Black-box policy contract for trading decision components."""

    def decide(self, strategy_input: StrategyInputT) -> StrategyOutputT:
        ...


@dataclass(frozen=True)
class PortfolioSizingTradingStrategy:
    sizing_policy: PortfolioSizingPolicy | None = None

    def decide(self, strategy_input: PortfolioDecisionInput) -> PortfolioDecisionOutput:
        return apply_portfolio_sizing_policy(
            strategy_input,
            sizing_policy=self.sizing_policy,
        )
