from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioPositionState:
    subject_id: str
    weight: float
    notional: float | None = None
    quantity: float | None = None


@dataclass(frozen=True)
class PortfolioState:
    portfolio_id: str | None = None
    as_of: str | None = None
    positions: tuple[PortfolioPositionState, ...] = ()
    capital_base: float = 1.0
    gross_limit: float | None = None
    net_limit: float | None = None
    rebalance_step: int = 0
    holding_period_days: int = 0
    recent_turnover: float = 0.0
    current_drawdown: float = 0.0

    @property
    def gross_exposure(self) -> float:
        return float(sum(abs(position.weight) for position in self.positions))

    @property
    def net_exposure(self) -> float:
        return float(sum(position.weight for position in self.positions))

    @property
    def weights_by_subject(self) -> dict[str, float]:
        return {
            position.subject_id: float(position.weight)
            for position in self.positions
        }
