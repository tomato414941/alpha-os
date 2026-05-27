from __future__ import annotations

from typing import Protocol

from .portfolio_decision import PortfolioDecisionInput, PortfolioDecisionOutput


class TradingStrategy(Protocol):
    """Black-box policy contract for trading decision components."""

    def decide(self, decision_input: PortfolioDecisionInput) -> PortfolioDecisionOutput:
        ...
