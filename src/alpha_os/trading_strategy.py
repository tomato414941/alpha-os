from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar


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
