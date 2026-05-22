from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contract_boundaries import (
    SubjectSetContractBoundary,
    default_subject_set_contract_boundary,
)


@dataclass(frozen=True)
class UniversePolicySpec:
    base_currency: str | None = None
    trading_calendar: str | None = None
    benchmark_id: str | None = None

    def to_document(self) -> dict[str, str | None]:
        return {
            "base_currency": self.base_currency,
            "trading_calendar": self.trading_calendar,
            "benchmark_id": self.benchmark_id,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "UniversePolicySpec":
        return cls(
            base_currency=(
                None
                if document.get("base_currency") is None
                else str(document.get("base_currency"))
            ),
            trading_calendar=(
                None
                if document.get("trading_calendar") is None
                else str(document.get("trading_calendar"))
            ),
            benchmark_id=(
                None
                if document.get("benchmark_id") is None
                else str(document.get("benchmark_id"))
            ),
        )


@dataclass(frozen=True)
class ObservationSpec:
    observation_spec_id: str
    observable_id: str
    adapter_kind: str = "signal_noise_asset_observable"
    source_id: str = "signal_noise"
    resolution: str = "1d"
    provided_observable_ids: tuple[str, ...] = ()
    research_price_observable_id: str | None = None
    tradable_price_observable_id: str | None = None
    metadata_observable_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class InstrumentSpec:
    instrument_id: str
    instrument_type: str
    asset: str
    venue: str | None = None
    quote_ccy: str | None = None
    collateral_ccy: str | None = None
    contract_family: str | None = None
    underlying_id: str | None = None
    asset_class: str | None = None
    region: str | None = None
    liquidity_tier: str | None = None
    cluster: str | None = None
    expiry: str | None = None
    roll_rule: str | None = None
    multiplier: float | None = None
    margin_model: str | None = None


@dataclass(frozen=True)
class SubjectObservationBinding:
    subject_id: str
    asset: str
    observation_spec_id: str
    subject_kind: str = "asset"
    instrument_id: str | None = None


@dataclass(frozen=True)
class SubjectSet:
    subject_set_id: str | None = None
    instruments: tuple[InstrumentSpec, ...] = ()
    observation_specs: tuple[ObservationSpec, ...] = ()
    bindings: tuple[SubjectObservationBinding, ...] = ()
    universe_policy: UniversePolicySpec = field(default_factory=UniversePolicySpec)

    def __post_init__(self) -> None:
        instrument_ids = [item.instrument_id for item in self.instruments]
        if len(instrument_ids) != len(set(instrument_ids)):
            raise ValueError("subject set contains duplicate instrument_id values")
        spec_ids = [item.observation_spec_id for item in self.observation_specs]
        if len(spec_ids) != len(set(spec_ids)):
            raise ValueError("subject set contains duplicate observation_spec_id values")
        subject_ids = [item.subject_id for item in self.bindings]
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError("subject set contains duplicate subject_id values")
        missing = [
            item.observation_spec_id
            for item in self.bindings
            if item.observation_spec_id not in self.observation_spec_by_id
        ]
        if missing:
            raise ValueError(
                "subject set binding references unknown observation_spec_id: "
                + ", ".join(sorted(set(missing)))
            )
        missing_instruments = [
            item.instrument_id
            for item in self.bindings
            if item.instrument_id is not None
            and item.instrument_id not in self.instrument_by_id
        ]
        if missing_instruments:
            raise ValueError(
                "subject set binding references unknown instrument_id: "
                + ", ".join(sorted(set(missing_instruments)))
            )

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return tuple(item.subject_id for item in self.bindings)

    @property
    def contract_boundary(self) -> SubjectSetContractBoundary:
        return default_subject_set_contract_boundary()

    @property
    def instrument_by_id(self) -> dict[str, InstrumentSpec]:
        return {
            item.instrument_id: item
            for item in self.instruments
        }

    @property
    def observation_spec_by_id(self) -> dict[str, ObservationSpec]:
        return {
            item.observation_spec_id: item
            for item in self.observation_specs
        }

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

    @property
    def observation_spec_id_by_subject(self) -> dict[str, str]:
        return {
            item.subject_id: item.observation_spec_id
            for item in self.bindings
        }

    @property
    def instrument_id_by_subject(self) -> dict[str, str]:
        return {
            item.subject_id: item.instrument_id
            for item in self.bindings
            if item.instrument_id is not None
        }

    def observation_spec_for_subject(self, subject_id: str) -> ObservationSpec:
        observation_spec_id = self.observation_spec_id_by_subject[subject_id]
        return self.observation_spec_by_id[observation_spec_id]

    def instrument_for_subject(self, subject_id: str) -> InstrumentSpec | None:
        instrument_id = self.instrument_id_by_subject.get(subject_id)
        if instrument_id is None:
            return None
        return self.instrument_by_id[instrument_id]

    @property
    def asset_class_by_subject(self) -> dict[str, str]:
        return {
            binding.subject_id: instrument.asset_class
            for binding in self.bindings
            if (instrument := self.instrument_for_subject(binding.subject_id)) is not None
            and instrument.asset_class is not None
        }

    @property
    def region_by_subject(self) -> dict[str, str]:
        return {
            binding.subject_id: instrument.region
            for binding in self.bindings
            if (instrument := self.instrument_for_subject(binding.subject_id)) is not None
            and instrument.region is not None
        }

    @property
    def liquidity_tier_by_subject(self) -> dict[str, str]:
        return {
            binding.subject_id: instrument.liquidity_tier
            for binding in self.bindings
            if (instrument := self.instrument_for_subject(binding.subject_id)) is not None
            and instrument.liquidity_tier is not None
        }

    @property
    def cluster_by_subject(self) -> dict[str, str]:
        return {
            binding.subject_id: instrument.cluster
            for binding in self.bindings
            if (instrument := self.instrument_for_subject(binding.subject_id)) is not None
            and instrument.cluster is not None
        }

    def subjects_grouped_by_instrument_field(
        self,
        field_name: str,
    ) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for binding in self.bindings:
            instrument = self.instrument_for_subject(binding.subject_id)
            if instrument is None:
                continue
            value = getattr(instrument, field_name, None)
            if value is None:
                continue
            grouped.setdefault(str(value), []).append(binding.subject_id)
        return {
            key: tuple(values)
            for key, values in grouped.items()
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
class RiskInput:
    name: str
    subject_id: str | None
    value: float
    horizon_days: int | None = None
    unit: str | None = None


@dataclass(frozen=True)
class CostInput:
    name: str
    subject_id: str | None
    value: float
    basis: str | None = None
    unit: str | None = None


@dataclass(frozen=True)
class UncertaintyInput:
    subject_id: str | None
    source_id: str
    target_id: str | None = None
    estimate_std: float = 0.0
    basis: str | None = None
    proxy_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelUncertaintyInput:
    subject_id: str | None
    source_id: str
    target_id: str | None = None
    model_error: float = 0.0
    basis: str | None = None
    proxy_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuralUncertaintyInput:
    subject_id: str | None
    source_id: str
    target_id: str | None = None
    structural_error: float = 0.0
    basis: str | None = None
    proxy_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DependenceInput:
    name: str
    left_subject_id: str
    right_subject_id: str
    value: float
    basis: str | None = None


@dataclass(frozen=True)
class HistoricalReturnInput:
    subject_id: str
    returns_by_date: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservedPortfolioInputs:
    predictive_signals: tuple[PredictiveSignalInput, ...] = ()
    risk_inputs: tuple[RiskInput, ...] = ()
    cost_inputs: tuple[CostInput, ...] = ()
    uncertainty_inputs: tuple[UncertaintyInput, ...] = ()
    model_uncertainty_inputs: tuple[ModelUncertaintyInput, ...] = ()
    structural_uncertainty_inputs: tuple[StructuralUncertaintyInput, ...] = ()
    dependence_inputs: tuple[DependenceInput, ...] = ()
    historical_return_inputs: tuple[HistoricalReturnInput, ...] = ()


@dataclass(frozen=True)
class PortfolioDecisionAssumptions:
    risk_inputs: tuple[RiskInput, ...] = ()
    cost_inputs: tuple[CostInput, ...] = ()
    uncertainty_inputs: tuple[UncertaintyInput, ...] = ()
    model_uncertainty_inputs: tuple[ModelUncertaintyInput, ...] = ()
    structural_uncertainty_inputs: tuple[StructuralUncertaintyInput, ...] = ()
    dependence_inputs: tuple[DependenceInput, ...] = ()


@dataclass(frozen=True)
class PortfolioDecisionInput:
    portfolio_id: str | None = None
    as_of: str | None = None
    portfolio_state: PortfolioState = field(default_factory=PortfolioState)
    observed_inputs: ObservedPortfolioInputs = field(default_factory=ObservedPortfolioInputs)
    assumptions: PortfolioDecisionAssumptions = field(default_factory=PortfolioDecisionAssumptions)
    subject_metadata_by_subject: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def predictive_signals(self) -> tuple[PredictiveSignalInput, ...]:
        return self.observed_inputs.predictive_signals

    @property
    def risk_inputs(self) -> tuple[RiskInput, ...]:
        return _merge_subject_inputs(
            self.observed_inputs.risk_inputs,
            self.assumptions.risk_inputs,
        )

    @property
    def cost_inputs(self) -> tuple[CostInput, ...]:
        return _merge_subject_inputs(
            self.observed_inputs.cost_inputs,
            self.assumptions.cost_inputs,
        )

    @property
    def uncertainty_inputs(self) -> tuple[UncertaintyInput, ...]:
        return _merge_uncertainty_inputs(
            self.observed_inputs.uncertainty_inputs,
            self.assumptions.uncertainty_inputs,
        )

    @property
    def model_uncertainty_inputs(self) -> tuple[ModelUncertaintyInput, ...]:
        return _merge_model_uncertainty_inputs(
            self.observed_inputs.model_uncertainty_inputs,
            self.assumptions.model_uncertainty_inputs,
        )

    @property
    def structural_uncertainty_inputs(self) -> tuple[StructuralUncertaintyInput, ...]:
        return _merge_structural_uncertainty_inputs(
            self.observed_inputs.structural_uncertainty_inputs,
            self.assumptions.structural_uncertainty_inputs,
        )

    @property
    def dependence_inputs(self) -> tuple[DependenceInput, ...]:
        return _merge_dependence_inputs(
            self.observed_inputs.dependence_inputs,
            self.assumptions.dependence_inputs,
        )


@dataclass(frozen=True)
class SizingRequest:
    subject_ids: tuple[str, ...]
    signal_values: tuple[float, ...]
    current_weights: tuple[float, ...]
    historical_return_matrix: tuple[tuple[float, ...], ...]
    asset_classes: tuple[str | None, ...]
    clusters: tuple[str | None, ...]
    uncertainty_std: tuple[float, ...]
    risk_values: tuple[float, ...]
    model_uncertainty_values: tuple[float, ...]
    structural_uncertainty_values: tuple[float, ...]
    dependence_values: tuple[float, ...]
    dependence_penalty_matrix: tuple[tuple[float, ...], ...]
    no_trade_bands: tuple[float, ...]
    market_impact_levels: tuple[float, ...]
    transaction_cost_levels: tuple[float, ...]
    short_cost_levels: tuple[float, ...]
    signal_horizons: tuple[int | None, ...]
    gross_exposure_cap: float | None
    net_exposure_cap: float | None
    capital_base: float
    holding_period_days: int
    current_drawdown: float
    recent_turnover: float
    turnover_friction: float


@dataclass(frozen=True)
class SizingDiagnostics:
    backend_id: str = "-"
    solver: str = "-"
    status: str = "-"
    objective_value: float | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True)
class SizingSolution:
    subject_ids: tuple[str, ...]
    target_weights: tuple[float, ...]
    risk_scales: tuple[float, ...]
    diagnostics: SizingDiagnostics = field(default_factory=SizingDiagnostics)


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
    portfolio_id: str | None = None
    as_of: str | None = None
    targets: tuple[PortfolioTarget, ...] = ()
    sizing_diagnostics: SizingDiagnostics = field(default_factory=SizingDiagnostics)

    @property
    def gross_target_exposure(self) -> float:
        return float(sum(abs(target.target_weight) for target in self.targets))

    @property
    def net_target_exposure(self) -> float:
        return float(sum(target.target_weight for target in self.targets))


def _merge_subject_inputs[T](
    observed: tuple[T, ...],
    assumptions: tuple[T, ...],
) -> tuple[T, ...]:
    merged: dict[tuple[str | None, str | None], T] = {}
    ordered_keys: list[tuple[str | None, str | None]] = []
    for item in observed:
        key = (
            getattr(item, "name", None),
            getattr(item, "subject_id", None),
        )
        if key not in merged:
            ordered_keys.append(key)
        merged[key] = item
    for item in assumptions:
        key = (
            getattr(item, "name", None),
            getattr(item, "subject_id", None),
        )
        if key not in merged:
            ordered_keys.append(key)
        merged[key] = item
    return tuple(merged[key] for key in ordered_keys)


def _merge_dependence_inputs(
    observed: tuple[DependenceInput, ...],
    assumptions: tuple[DependenceInput, ...],
) -> tuple[DependenceInput, ...]:
    merged: dict[tuple[str, str, str], DependenceInput] = {}
    ordered_keys: list[tuple[str, str, str]] = []
    for item in observed:
        key = (item.name, item.left_subject_id, item.right_subject_id)
        if key not in merged:
            ordered_keys.append(key)
        merged[key] = item
    for item in assumptions:
        key = (item.name, item.left_subject_id, item.right_subject_id)
        if key not in merged:
            ordered_keys.append(key)
        merged[key] = item
    return tuple(merged[key] for key in ordered_keys)


def _merge_uncertainty_inputs(
    observed: tuple[UncertaintyInput, ...],
    assumptions: tuple[UncertaintyInput, ...],
) -> tuple[UncertaintyInput, ...]:
    merged: dict[tuple[str | None, str, str | None], UncertaintyInput] = {}
    ordered_keys: list[tuple[str | None, str, str | None]] = []
    for item in observed:
        key = (item.subject_id, item.source_id, item.target_id)
        if key not in merged:
            ordered_keys.append(key)
        merged[key] = item
    for item in assumptions:
        key = (item.subject_id, item.source_id, item.target_id)
        if key not in merged:
            ordered_keys.append(key)
        merged[key] = item
    return tuple(merged[key] for key in ordered_keys)


def _merge_model_uncertainty_inputs(
    observed: tuple[ModelUncertaintyInput, ...],
    assumptions: tuple[ModelUncertaintyInput, ...],
) -> tuple[ModelUncertaintyInput, ...]:
    merged: dict[tuple[str | None, str, str | None], ModelUncertaintyInput] = {}
    ordered_keys: list[tuple[str | None, str, str | None]] = []
    for item in observed:
        key = (item.subject_id, item.source_id, item.target_id)
        if key not in merged:
            ordered_keys.append(key)
        merged[key] = item
    for item in assumptions:
        key = (item.subject_id, item.source_id, item.target_id)
        if key not in merged:
            ordered_keys.append(key)
        merged[key] = item
    return tuple(merged[key] for key in ordered_keys)


def _merge_structural_uncertainty_inputs(
    observed: tuple[StructuralUncertaintyInput, ...],
    assumptions: tuple[StructuralUncertaintyInput, ...],
) -> tuple[StructuralUncertaintyInput, ...]:
    merged: dict[tuple[str | None, str, str | None], StructuralUncertaintyInput] = {}
    ordered_keys: list[tuple[str | None, str, str | None]] = []
    for item in observed:
        key = (item.subject_id, item.source_id, item.target_id)
        if key not in merged:
            ordered_keys.append(key)
        merged[key] = item
    for item in assumptions:
        key = (item.subject_id, item.source_id, item.target_id)
        if key not in merged:
            ordered_keys.append(key)
        merged[key] = item
    return tuple(merged[key] for key in ordered_keys)
