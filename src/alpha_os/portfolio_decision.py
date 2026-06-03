from __future__ import annotations

from dataclasses import dataclass

from .contract_boundaries import (
    SubjectSetContractBoundary,
    default_subject_set_contract_boundary,
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
