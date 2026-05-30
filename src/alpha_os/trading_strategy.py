from __future__ import annotations

from typing import Protocol, TypeVar


StrategyInputT = TypeVar("StrategyInputT", contravariant=True)
StrategyOutputT = TypeVar("StrategyOutputT", covariant=True)


class TradingStrategy(Protocol[StrategyInputT, StrategyOutputT]):
    """Black-box policy contract for trading decision components."""

    def decide(self, strategy_input: StrategyInputT) -> StrategyOutputT:
        ...
