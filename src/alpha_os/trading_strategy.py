from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TradingStrategyInput:
    pass


@dataclass(frozen=True)
class TradingStrategyOutput:
    pass


class TradingStrategy(Protocol):
    """Black-box policy contract for trading decision components."""

    def decide(self, strategy_input: TradingStrategyInput) -> TradingStrategyOutput:
        ...

