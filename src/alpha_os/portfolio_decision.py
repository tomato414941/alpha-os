from __future__ import annotations

from dataclasses import dataclass, field

from .config import DEFAULT_ASSET


@dataclass(frozen=True)
class PortfolioPositionState:
    subject_id: str
    weight: float
    notional: float | None = None
    quantity: float | None = None


@dataclass(frozen=True)
class PortfolioState:
    # `asset` is a bounded-runtime convenience field.
    # The durable allocation unit is `subject_id` inside positions.
    asset: str = DEFAULT_ASSET
    as_of: str | None = None
    positions: tuple[PortfolioPositionState, ...] = ()

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


@dataclass(frozen=True)
class PredictiveSignalInput:
    source_id: str
    subject_id: str
    target_id: str
    value: float
    confidence: float | None = None
    source_kind: str | None = None


@dataclass(frozen=True)
class PortfolioScalarInput:
    name: str
    subject_id: str | None
    value: float


@dataclass(frozen=True)
class PortfolioDecisionInput:
    # `asset` remains a bounded-runtime convenience field.
    # Long-run decision inputs should be keyed by `subject_id`.
    asset: str = DEFAULT_ASSET
    as_of: str | None = None
    portfolio_state: PortfolioState = field(default_factory=PortfolioState)
    predictive_signals: tuple[PredictiveSignalInput, ...] = ()
    risk_inputs: tuple[PortfolioScalarInput, ...] = ()
    cost_inputs: tuple[PortfolioScalarInput, ...] = ()
    uncertainty_inputs: tuple[PortfolioScalarInput, ...] = ()
    dependence_inputs: tuple[PortfolioScalarInput, ...] = ()


@dataclass(frozen=True)
class PortfolioTarget:
    subject_id: str
    target_weight: float
    position_delta: float
    target_notional: float | None = None
    target_quantity: float | None = None
    entry_allowed: bool = True
    risk_scale: float = 1.0


@dataclass(frozen=True)
class PortfolioDecisionOutput:
    # `asset` remains a bounded-runtime convenience field.
    # The durable output unit is `subject_id` inside targets.
    asset: str = DEFAULT_ASSET
    as_of: str | None = None
    targets: tuple[PortfolioTarget, ...] = ()

    @property
    def gross_target_exposure(self) -> float:
        return float(sum(abs(target.target_weight) for target in self.targets))

    @property
    def net_target_exposure(self) -> float:
        return float(sum(target.target_weight for target in self.targets))
