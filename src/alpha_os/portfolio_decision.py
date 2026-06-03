from __future__ import annotations

from dataclasses import dataclass

from .contract_boundaries import (
    SubjectSetContractBoundary,
    default_subject_set_contract_boundary,
)


@dataclass(frozen=True)
class SubjectObservationBinding:
    subject_id: str
    asset: str
    subject_kind: str = "asset"


@dataclass(frozen=True)
class SubjectSet:
    subject_set_id: str | None = None
    bindings: tuple[SubjectObservationBinding, ...] = ()

    def __post_init__(self) -> None:
        subject_ids = [item.subject_id for item in self.bindings]
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError("subject set contains duplicate subject_id values")

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return tuple(item.subject_id for item in self.bindings)

    @property
    def contract_boundary(self) -> SubjectSetContractBoundary:
        return default_subject_set_contract_boundary()

    @property
    def asset_by_subject(self) -> dict[str, str]:
        return {
            item.subject_id: item.asset
            for item in self.bindings
        }

    @property
    def subject_kind_by_subject(self) -> dict[str, str]:
        return {
            item.subject_id: item.subject_kind
            for item in self.bindings
        }


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
