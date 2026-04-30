from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import DEFAULT_SUBJECT_ID, DEFAULT_TARGET, default_runtime_asset
from .compression import CompressedBelief
from .cross_instrument_contract import (
    CrossInstrumentReportContract,
    default_validation_result_set_cross_instrument_contract,
)
from .evaluation_task import EvaluationTask
from .evaluation_job_spec import EvaluationJobSpec
from .evaluation_spec import EvaluationSpec
from .evaluation_report import EvaluationReport
from .signal_registry import (
    SignalDefinition,
    SignalSpec,
    find_signal_definition,
    find_signal_spec,
    subject_id_for_signal,
)
from .signal_discovery import SignalDiscoverySpec
from .observables import (
    ObservableDefinition,
    find_observable_definition,
    list_observable_definitions,
)
from .strategy_adaptation import StrategyAdaptationState
from .portfolio_decision import (
    InstrumentSpec,
    ObservationSpec,
    SubjectObservationBinding,
    SubjectSet,
    UniversePolicySpec,
)
from .screening import ScreeningResult
from .signal_discovery_run import SignalDiscoveryRun
from .trading_strategy import TradingStrategySpec
from .targets import TargetDefinition, find_target_definition, list_target_definitions
from .initial_strategy_state import InitialStrategyState
from .transition_policy import decide_operator_transition
from .validation_result_set import ValidationResultSet


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, init=False)
class SignalState:
    signal_id: str
    signal_spec_id: str | None
    subject_id: str
    asset: str
    target_id: str
    definition_json: str | None
    status: str
    prediction_count: int
    observation_count: int

    def __init__(
        self,
        *,
        signal_id: str | None = None,

        signal_spec_id: str | None = None,
        specification_signal_id: str | None = None,
        subject_id: str,
        asset: str,
        target_id: str,
        definition_json: str | None,
        status: str,
        prediction_count: int,
        observation_count: int,
    ) -> None:
        resolved_signal_id = signal_id
        if resolved_signal_id is None:
            raise ValueError("signal state requires signal_id")
        object.__setattr__(self, "signal_id", str(resolved_signal_id))
        object.__setattr__(
            self,
            "signal_spec_id",
            (
                signal_spec_id
                if signal_spec_id is not None
                else specification_signal_id
            ),
        )
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "definition_json", definition_json)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "prediction_count", prediction_count)
        object.__setattr__(self, "observation_count", observation_count)
    @property
    def specification_signal_id(self) -> str | None:
        return self.signal_spec_id

    @property
    def definition(self) -> dict[str, Any] | None:
        if self.definition_json is None:
            return None
        return json.loads(self.definition_json)

    @property
    def target_definition(self) -> TargetDefinition | None:
        definition = self.definition
        if definition is None:
            return None
        target_document = definition.get("target_definition")
        if not isinstance(target_document, dict):
            return None
        return TargetDefinition.from_document(target_document)

    @property
    def kind(self) -> str | None:
        definition = self.definition
        if definition is None:
            return None
        value = definition.get("kind")
        return value if isinstance(value, str) else None

    @property
    def observation_spec(self) -> ObservationSpec | None:
        definition = self.definition
        if definition is None:
            return None
        observation_spec_document = definition.get("observation_spec")
        if isinstance(observation_spec_document, dict):
            return ObservationSpec(
                observation_spec_id=str(
                    observation_spec_document.get(
                        "observation_spec_id",
                        f"{self.signal_id}__observation",
                    )
                ),
                observable_id=str(
                    observation_spec_document.get("observable_id", "daily_close")
                ),
                adapter_kind=str(
                    observation_spec_document.get(
                        "adapter_kind",
                        "signal_noise_asset_observable",
                    )
                ),
                source_id=str(observation_spec_document.get("source_id", "signal_noise")),
                resolution=str(observation_spec_document.get("resolution", "1d")),
                provided_observable_ids=tuple(
                    str(value)
                    for value in observation_spec_document.get(
                        "provided_observable_ids",
                        [],
                    )
                    if isinstance(value, str) and value
                ),
            )
        return None

    @property
    def signal_name(self) -> str | None:
        definition = self.definition
        if definition is None:
            return None
        value = definition.get("signal_name")
        return value if isinstance(value, str) and value else None

    @property
    def observation_text(self) -> str | None:
        observation_spec = self.observation_spec
        if observation_spec is None:
            return None
        return (
            f"{observation_spec.observable_id}@"
            f"{observation_spec.adapter_kind}:"
            f"{observation_spec.source_id}/"
            f"{observation_spec.resolution}"
        )

    @property
    def lookback(self) -> int | None:
        definition = self.definition
        if definition is None:
            return None
        params = definition.get("params")
        if not isinstance(params, dict):
            return None
        value = params.get("lookback")
        return value if isinstance(value, int) else None

    @property
    def horizon_days(self) -> int | None:
        target_definition = self.target_definition
        if target_definition is None:
            return None
        return target_definition.horizon_days


@dataclass(frozen=True, init=False)
class SignalSpecState:
    signal_id: str
    definition_json: str
    target_id: str

    def __init__(
        self,
        *,
        signal_id: str | None = None,

        definition_json: str,
        target_id: str,
    ) -> None:
        resolved_signal_id = signal_id
        if resolved_signal_id is None:
            raise ValueError("signal spec state requires signal_id")
        object.__setattr__(self, "signal_id", str(resolved_signal_id))
        object.__setattr__(self, "definition_json", definition_json)
        object.__setattr__(self, "target_id", target_id)
    @property
    def definition(self) -> SignalSpec:
        return SignalSpec.from_document(
            signal_id=self.signal_id,
            document=json.loads(self.definition_json),
        )

    @property
    def kind(self) -> str:
        return self.definition.kind

    @property
    def lookback(self) -> int:
        return self.definition.lookback

    @property
    def horizon_days(self) -> int | None:
        return self.definition.horizon_days


@dataclass(frozen=True, init=False)
class EvaluationSnapshot:
    evaluation_id: str
    subject_id: str
    asset: str
    target_id: str
    signal_id: str
    prediction_value: float
    observation_value: float
    signed_edge: float
    absolute_error: float
    input_source: str | None
    input_range_start: str | None
    input_range_end: str | None
    funding_cost_bps: float | None
    borrow_fee_bps: float | None
    roll_cost_bps: float | None
    financing_cost_bps: float | None
    contract_multiplier: float | None
    contract_id: str | None
    contract_family: str | None
    quote_ccy: str | None
    collateral_ccy: str | None
    roll_event: dict[str, object] | None
    observation_spec_id: str | None
    observable_id: str | None
    adapter_kind: str | None
    created_at: str

    def __init__(
        self,
        *,
        evaluation_id: str,
        subject_id: str,
        asset: str,
        target_id: str,

        signal_id: str | None = None,
        prediction_value: float,
        observation_value: float,
        signed_edge: float,
        absolute_error: float,
        input_source: str | None,
        input_range_start: str | None,
        input_range_end: str | None,
        funding_cost_bps: float | None = None,
        borrow_fee_bps: float | None = None,
        roll_cost_bps: float | None = None,
        financing_cost_bps: float | None = None,
        contract_multiplier: float | None = None,
        contract_id: str | None = None,
        contract_family: str | None = None,
        quote_ccy: str | None = None,
        collateral_ccy: str | None = None,
        roll_event: dict[str, object] | None = None,
        observation_spec_id: str | None = None,
        observable_id: str | None,
        adapter_kind: str | None,
        created_at: str,
    ) -> None:
        resolved_signal_id = signal_id
        if resolved_signal_id is None:
            raise ValueError("evaluation snapshot requires signal_id")
        object.__setattr__(self, "evaluation_id", evaluation_id)
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "signal_id", str(resolved_signal_id))
        object.__setattr__(self, "prediction_value", prediction_value)
        object.__setattr__(self, "observation_value", observation_value)
        object.__setattr__(self, "signed_edge", signed_edge)
        object.__setattr__(self, "absolute_error", absolute_error)
        object.__setattr__(self, "input_source", input_source)
        object.__setattr__(self, "input_range_start", input_range_start)
        object.__setattr__(self, "input_range_end", input_range_end)
        object.__setattr__(self, "funding_cost_bps", funding_cost_bps)
        object.__setattr__(self, "borrow_fee_bps", borrow_fee_bps)
        object.__setattr__(self, "roll_cost_bps", roll_cost_bps)
        object.__setattr__(self, "financing_cost_bps", financing_cost_bps)
        object.__setattr__(self, "contract_multiplier", contract_multiplier)
        object.__setattr__(self, "contract_id", contract_id)
        object.__setattr__(self, "contract_family", contract_family)
        object.__setattr__(self, "quote_ccy", quote_ccy)
        object.__setattr__(self, "collateral_ccy", collateral_ccy)
        object.__setattr__(self, "roll_event", roll_event)
        object.__setattr__(self, "observation_spec_id", observation_spec_id)
        object.__setattr__(self, "observable_id", observable_id)
        object.__setattr__(self, "adapter_kind", adapter_kind)
        object.__setattr__(self, "created_at", created_at)

@dataclass(frozen=True, init=False)
class PredictionRecord:
    evaluation_id: str
    signal_id: str
    subject_id: str
    asset: str
    target_id: str
    value: float
    recorded_at: str

    def __init__(
        self,
        *,
        evaluation_id: str,

        signal_id: str | None = None,
        subject_id: str,
        asset: str,
        target_id: str,
        value: float,
        recorded_at: str,
    ) -> None:
        resolved_signal_id = signal_id
        if resolved_signal_id is None:
            raise ValueError("prediction record requires signal_id")
        object.__setattr__(self, "evaluation_id", evaluation_id)
        object.__setattr__(self, "signal_id", str(resolved_signal_id))
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "recorded_at", recorded_at)

@dataclass(frozen=True)
class ObservationRecord:
    evaluation_id: str
    subject_id: str
    asset: str
    target_id: str
    value: float
    recorded_at: str


@dataclass(frozen=True)
class TargetState:
    target_id: str
    definition_json: str

    @property
    def definition(self) -> TargetDefinition:
        return TargetDefinition.from_document(json.loads(self.definition_json))


@dataclass(frozen=True)
class ObservableState:
    observable_id: str
    definition_json: str

    @property
    def definition(self) -> ObservableDefinition:
        return ObservableDefinition.from_document(json.loads(self.definition_json))


@dataclass(frozen=True)
class SubjectSetState:
    subject_set_id: str
    definition_json: str

    @property
    def definition(self) -> SubjectSet:
        document = json.loads(self.definition_json)
        instruments = tuple(
            InstrumentSpec(
                instrument_id=str(item["instrument_id"]),
                instrument_type=str(item["instrument_type"]),
                asset=str(item["asset"]),
                venue=None if item.get("venue") is None else str(item.get("venue")),
                quote_ccy=(
                    None if item.get("quote_ccy") is None else str(item.get("quote_ccy"))
                ),
                collateral_ccy=(
                    None
                    if item.get("collateral_ccy") is None
                    else str(item.get("collateral_ccy"))
                ),
                contract_family=(
                    None
                    if item.get("contract_family") is None
                    else str(item.get("contract_family"))
                ),
                underlying_id=(
                    None
                    if item.get("underlying_id") is None
                    else str(item.get("underlying_id"))
                ),
                asset_class=(
                    None if item.get("asset_class") is None else str(item.get("asset_class"))
                ),
                region=(
                    None if item.get("region") is None else str(item.get("region"))
                ),
                liquidity_tier=(
                    None
                    if item.get("liquidity_tier") is None
                    else str(item.get("liquidity_tier"))
                ),
                cluster=(
                    None if item.get("cluster") is None else str(item.get("cluster"))
                ),
                expiry=None if item.get("expiry") is None else str(item.get("expiry")),
                roll_rule=(
                    None if item.get("roll_rule") is None else str(item.get("roll_rule"))
                ),
                multiplier=(
                    None if item.get("multiplier") is None else float(item.get("multiplier"))
                ),
                margin_model=(
                    None if item.get("margin_model") is None else str(item.get("margin_model"))
                ),
            )
            for item in document.get("instruments", [])
            if isinstance(item, dict)
            and isinstance(item.get("instrument_id"), str)
            and isinstance(item.get("instrument_type"), str)
            and isinstance(item.get("asset"), str)
        )
        observation_specs = tuple(
            ObservationSpec(
                observation_spec_id=str(item["observation_spec_id"]),
                observable_id=str(item["observable_id"]),
                adapter_kind=str(
                    item.get("adapter_kind", "signal_noise_asset_observable")
                ),
                source_id=str(item.get("source_id", "signal_noise")),
                resolution=str(item.get("resolution", "1d")),
                provided_observable_ids=tuple(
                    str(value)
                    for value in item.get("provided_observable_ids", [])
                    if isinstance(value, str) and value
                ),
                research_price_observable_id=(
                    None
                    if item.get("research_price_observable_id") is None
                    else str(item.get("research_price_observable_id"))
                ),
                tradable_price_observable_id=(
                    None
                    if item.get("tradable_price_observable_id") is None
                    else str(item.get("tradable_price_observable_id"))
                ),
                metadata_observable_ids=tuple(
                    str(value)
                    for value in item.get("metadata_observable_ids", [])
                    if isinstance(value, str) and value
                ),
            )
            for item in document.get("observation_specs", [])
            if isinstance(item, dict)
            and isinstance(item.get("observation_spec_id"), str)
            and isinstance(item.get("observable_id"), str)
        )
        bindings = tuple(
            SubjectObservationBinding(
                subject_id=str(item["subject_id"]),
                asset=str(item["asset"]),
                observation_spec_id=str(item["observation_spec_id"]),
                subject_kind=str(item.get("subject_kind", "asset")),
                instrument_id=(
                    None if item.get("instrument_id") is None else str(item.get("instrument_id"))
                ),
            )
            for item in document.get("bindings", [])
            if isinstance(item, dict)
            and isinstance(item.get("subject_id"), str)
            and isinstance(item.get("asset"), str)
            and isinstance(item.get("observation_spec_id"), str)
        )
        if not observation_specs and bindings:
            legacy_bindings = tuple(
                item
                for item in document.get("bindings", [])
                if isinstance(item, dict)
                and isinstance(item.get("subject_id"), str)
                and isinstance(item.get("asset"), str)
                and isinstance(item.get("signal_name"), str)
            )
            observation_specs = tuple(
                ObservationSpec(
                    observation_spec_id=f"{str(item['subject_id'])}__legacy",
                    observable_id="daily_close",
                    adapter_kind="signal_noise_asset_observable",
                )
                for item in legacy_bindings
            )
            bindings = tuple(
                SubjectObservationBinding(
                    subject_id=str(item["subject_id"]),
                    asset=str(item["asset"]),
                    observation_spec_id=f"{str(item['subject_id'])}__legacy",
                    subject_kind="asset",
                )
                for item in legacy_bindings
            )
        return SubjectSet(
            subject_set_id=self.subject_set_id,
            instruments=instruments,
            observation_specs=observation_specs,
            bindings=bindings,
            universe_policy=UniversePolicySpec(
                base_currency=(
                    None
                    if not isinstance(document.get("universe_policy"), dict)
                    or document["universe_policy"].get("base_currency") is None
                    else str(document["universe_policy"].get("base_currency"))
                ),
                trading_calendar=(
                    None
                    if not isinstance(document.get("universe_policy"), dict)
                    or document["universe_policy"].get("trading_calendar") is None
                    else str(document["universe_policy"].get("trading_calendar"))
                ),
                benchmark_id=(
                    None
                    if not isinstance(document.get("universe_policy"), dict)
                    or document["universe_policy"].get("benchmark_id") is None
                    else str(document["universe_policy"].get("benchmark_id"))
                ),
            ),
        )


@dataclass(frozen=True)
class SignalDiscoverySpecState:
    signal_discovery_id: str
    definition_json: str

    @property
    def definition(self) -> SignalDiscoverySpec:
        return SignalDiscoverySpec.from_document(
            signal_discovery_id=self.signal_discovery_id,
            document=json.loads(self.definition_json),
        )


@dataclass(frozen=True)
class EvaluationSpecState:
    evaluation_spec_id: str
    definition_json: str

    @property
    def definition(self) -> EvaluationSpec:
        return EvaluationSpec.from_document(json.loads(self.definition_json))


@dataclass(frozen=True, init=False)
class SignalMetricState:
    signal_id: str
    corr: float
    mmc: float | None
    mmc_baseline_type: str | None
    mmc_peer_count: int
    sample_count: int
    mmc_sample_count: int
    window_size: int
    start_evaluation_id: str | None
    end_evaluation_id: str | None
    updated_at: str

    def __init__(
        self,
        *,
        signal_id: str | None = None,

        corr: float,
        mmc: float | None,
        mmc_baseline_type: str | None,
        mmc_peer_count: int,
        sample_count: int,
        mmc_sample_count: int,
        window_size: int,
        start_evaluation_id: str | None,
        end_evaluation_id: str | None,
        updated_at: str,
    ) -> None:
        resolved_signal_id = signal_id
        if resolved_signal_id is None:
            raise ValueError("signal metric state requires signal_id")
        object.__setattr__(self, "signal_id", str(resolved_signal_id))
        object.__setattr__(self, "corr", corr)
        object.__setattr__(self, "mmc", mmc)
        object.__setattr__(self, "mmc_baseline_type", mmc_baseline_type)
        object.__setattr__(self, "mmc_peer_count", mmc_peer_count)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "mmc_sample_count", mmc_sample_count)
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(self, "start_evaluation_id", start_evaluation_id)
        object.__setattr__(self, "end_evaluation_id", end_evaluation_id)
        object.__setattr__(self, "updated_at", updated_at)

@dataclass(frozen=True)
class MetaPredictionState:
    evaluation_id: str
    subject_id: str
    asset: str
    target_id: str
    aggregation_kind: str
    value: float
    contributor_count: int
    details_json: str | None
    created_at: str
    updated_at: str

    @property
    def details(self) -> dict[str, Any] | None:
        if self.details_json is None:
            return None
        return json.loads(self.details_json)


@dataclass(frozen=True)
class MetaPredictionMetricState:
    aggregation_kind: str
    subject_id: str
    asset: str
    target_id: str
    corr: float
    sample_count: int
    window_size: int
    start_evaluation_id: str | None
    end_evaluation_id: str | None
    updated_at: str


@dataclass(frozen=True)
class ValidationRunState:
    run_id: str
    spec_json: str
    created_at: str
    criteria_json: str | None = None
    summary_json: str | None = None

    @property
    def cross_instrument_contract(self) -> CrossInstrumentReportContract:
        if self.criteria_json is None:
            return default_validation_result_set_cross_instrument_contract()
        return CrossInstrumentReportContract.from_document(json.loads(self.criteria_json))

    @property
    def validation_result_set(self) -> ValidationResultSet | None:
        if self.summary_json is None:
            return None
        return ValidationResultSet.from_document(json.loads(self.summary_json))


@dataclass(frozen=True)
class ScreeningResultState:
    screening_result_id: str
    signal_discovery_id: str
    result_json: str
    created_at: str

    @property
    def result(self) -> ScreeningResult:
        return ScreeningResult.from_document(
            screening_result_id=self.screening_result_id,
            document=json.loads(self.result_json),
        )


@dataclass(frozen=True)
class CompressedBeliefState:
    compressed_belief_id: str
    signal_discovery_id: str
    belief_json: str
    created_at: str

    @property
    def belief(self) -> CompressedBelief:
        return CompressedBelief.from_document(
            compressed_belief_id=self.compressed_belief_id,
            document=json.loads(self.belief_json),
        )


@dataclass(frozen=True)
class SignalDiscoveryRunState:
    signal_discovery_run_id: str
    signal_discovery_id: str
    run_json: str
    created_at: str

    @property
    def run(self) -> SignalDiscoveryRun:
        return SignalDiscoveryRun.from_document(
            signal_discovery_run_id=self.signal_discovery_run_id,
            document=json.loads(self.run_json),
        )


@dataclass(frozen=True)
class InitialStrategyStateRecord:
    initial_strategy_state_id: str
    strategy_id: str
    signal_train_id: str
    signal_discovery_id: str | None
    artifact_json: str
    created_at: str

    @property
    def state(self) -> InitialStrategyState:
        return InitialStrategyState.from_document(
            initial_strategy_state_id=self.initial_strategy_state_id,
            document=json.loads(self.artifact_json),
        )


@dataclass(frozen=True)
class TradingStrategyState:
    strategy_id: str
    spec_json: str
    created_at: str

    @property
    def trading_strategy(self) -> TradingStrategySpec:
        return TradingStrategySpec.from_document(
            json.loads(self.spec_json)["trading_strategy"]
        )


@dataclass(frozen=True)
class EvaluationTaskState:
    evaluation_task_id: str
    task_json: str
    created_at: str

    @property
    def task(self) -> EvaluationTask:
        return EvaluationTask.from_document(
            evaluation_task_id=self.evaluation_task_id,
            document=json.loads(self.task_json),
        )


@dataclass(frozen=True)
class EvaluationJobSpecState:
    evaluation_task_id: str
    job_spec_json: str
    created_at: str

    @property
    def job_spec(self) -> EvaluationJobSpec:
        return EvaluationJobSpec.from_document(json.loads(self.job_spec_json))


@dataclass(frozen=True)
class EvaluationReportState:
    evaluation_report_id: str
    evaluation_spec_id: str
    report_json: str
    created_at: str

    @property
    def report(self) -> EvaluationReport:
        return EvaluationReport.from_document(
            evaluation_report_id=self.evaluation_report_id,
            document=json.loads(self.report_json),
        )


@dataclass(frozen=True)
class StrategyAdaptationStateRecord:
    strategy_id: str
    signal_discovery_id: str | None
    state_json: str
    created_at: str

    @property
    def state(self) -> StrategyAdaptationState:
        return StrategyAdaptationState.from_document(
            strategy_id=self.strategy_id,
            document=json.loads(self.state_json),
        )


@dataclass(frozen=True, init=False)
class ValidationSignalResultState:
    run_id: str
    date_range_label: str
    start_date: str
    end_date: str
    target_id: str
    signal_id: str
    window_size: int
    corr: float
    mmc: float | None
    sample_count: int
    mmc_sample_count: int
    mmc_peer_count: int
    mmc_baseline_type: str | None
    updated_at: str

    def __init__(
        self,
        *,
        run_id: str,
        date_range_label: str,
        start_date: str,
        end_date: str,
        target_id: str,

        signal_id: str | None = None,
        window_size: int,
        corr: float,
        mmc: float | None,
        sample_count: int,
        mmc_sample_count: int,
        mmc_peer_count: int,
        mmc_baseline_type: str | None,
        updated_at: str,
    ) -> None:
        resolved_signal_id = signal_id
        if resolved_signal_id is None:
            raise ValueError("validation result requires signal_id")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "date_range_label", date_range_label)
        object.__setattr__(self, "start_date", start_date)
        object.__setattr__(self, "end_date", end_date)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "signal_id", str(resolved_signal_id))
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(self, "corr", corr)
        object.__setattr__(self, "mmc", mmc)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "mmc_sample_count", mmc_sample_count)
        object.__setattr__(self, "mmc_peer_count", mmc_peer_count)
        object.__setattr__(self, "mmc_baseline_type", mmc_baseline_type)
        object.__setattr__(self, "updated_at", updated_at)

@dataclass(frozen=True)
class ValidationMetaResultState:
    run_id: str
    date_range_label: str
    start_date: str
    end_date: str
    target_id: str
    aggregation_kind: str
    window_size: int
    corr: float
    sample_count: int
    updated_at: str


@dataclass(frozen=True)
class ValidationDecisionResultState:
    run_id: str
    date_range_label: str
    start_date: str
    end_date: str
    target_id: str
    subject_set_id: str
    aggregation_kind: str
    window_size: int
    gross_return_total: float
    net_return_total: float
    max_drawdown: float
    mean_turnover: float
    mean_gross_notional_exposure: float
    mean_net_notional_exposure: float
    mean_long_notional_exposure: float
    mean_short_notional_exposure: float
    mean_traded_notional: float
    cost_notional_total: float
    funding_cost_notional_total: float
    borrow_cost_notional_total: float
    roll_cost_notional_total: float
    step_count: int
    updated_at: str


@dataclass(frozen=True)
class PortfolioDecisionState:
    portfolio_id: str
    subject_id: str
    target_id: str
    aggregation_kind: str
    as_of: str
    target_weight: float
    position_delta: float
    target_notional: float | None
    target_quantity: float | None
    entry_allowed: bool
    risk_scale: float
    details_json: str | None
    created_at: str
    updated_at: str

    @property
    def details(self) -> dict[str, Any] | None:
        if self.details_json is None:
            return None
        return json.loads(self.details_json)


@dataclass(frozen=True)
class EvaluationDecisionTraceStepState:
    evaluation_report_id: str
    evaluation_task_id: str
    evaluation_fold_label: str
    evaluation_range_label: str
    variant: str
    step_index: int
    step_as_of: str
    step_granularity: str
    target_id: str
    subject_set_id: str
    gross_return: float
    net_return: float
    gross_pnl_notional: float
    net_pnl_notional: float
    turnover: float
    traded_notional: float
    cost_notional: float
    funding_cost_notional: float
    borrow_cost_notional: float
    roll_cost_notional: float

    gross_leverage_exposure: float
    net_leverage_exposure: float
    long_leverage_exposure: float
    short_leverage_exposure: float
    gross_notional_exposure: float
    net_notional_exposure: float
    long_notional_exposure: float
    short_notional_exposure: float
    gross_equity: float
    net_equity: float
    created_at: str


@dataclass(frozen=True)
class EvaluationDecisionTraceSubjectStepState:
    evaluation_report_id: str
    evaluation_task_id: str
    evaluation_fold_label: str
    evaluation_range_label: str
    variant: str
    step_index: int
    subject_id: str
    asset_class: str | None
    cluster: str | None
    signal_value: float
    realized_return: float
    target_weight: float
    position_delta: float
    target_notional: float
    traded_notional: float
    gross_pnl_notional: float
    execution_cost_notional: float
    funding_cost_notional: float
    borrow_cost_notional: float
    roll_cost_notional: float
    cost_notional: float
    net_pnl_notional: float
    net_return_contribution: float
    risk_scale: float
    entry_allowed: bool
    funding_cost_bps: float
    borrow_fee_bps: float
    roll_cost_bps: float
    contract_multiplier: float | None
    target_contracts: float | None
    traded_contracts: float | None
    created_at: str


SignalMetricState = SignalMetricState
ValidationSignalResultState = ValidationSignalResultState


def _row_to_signal(
    row: sqlite3.Row | None,
) -> SignalState | None:
    if row is None:
        return None
    subject_id = (
        str(row["subject_id"])
        if "subject_id" in row.keys() and row["subject_id"] is not None
        else subject_id_for_signal(
            signal_id=str(row["signal_id"]),
            asset=str(row["asset"]),
        )
    )
    return SignalState(
        signal_id=str(row["signal_id"]),
        signal_spec_id=(
            None
            if "specification_signal_id" not in row.keys()
            or row["specification_signal_id"] is None
            else str(row["specification_signal_id"])
        ),
        subject_id=subject_id,
        asset=str(row["asset"]),
        target_id=str(row["target_id"]),
        definition_json=None
        if row["definition_json"] is None
        else str(row["definition_json"]),
        status=str(row["status"]),
        prediction_count=int(row["prediction_count"]),
        observation_count=int(row["observation_count"]),
    )


def _row_to_signal_spec(
    row: sqlite3.Row | None,
) -> SignalSpecState | None:
    if row is None:
        return None
    return SignalSpecState(
        signal_id=str(row["signal_id"]),
        definition_json=str(row["definition_json"]),
        target_id=str(row["target_id"]),
    )


def _row_to_signal_discovery_spec(
    row: sqlite3.Row | None,
) -> SignalDiscoverySpecState | None:
    if row is None:
        return None
    return SignalDiscoverySpecState(
        signal_discovery_id=str(row["signal_discovery_id"]),
        definition_json=str(row["definition_json"]),
    )


def _row_to_evaluation_spec(
    row: sqlite3.Row | None,
) -> EvaluationSpecState | None:
    if row is None:
        return None
    return EvaluationSpecState(
        evaluation_spec_id=str(row["evaluation_spec_id"]),
        definition_json=str(row["definition_json"]),
    )


def _row_to_snapshot(row: sqlite3.Row | None) -> EvaluationSnapshot | None:
    if row is None:
        return None
    observation_spec_id = (
        None
        if "observation_spec_id" not in row.keys() or row["observation_spec_id"] is None
        else str(row["observation_spec_id"])
    )
    observable_id = (
        None
        if "observable_id" not in row.keys() or row["observable_id"] is None
        else str(row["observable_id"])
    )
    adapter_kind = (
        None
        if "adapter_kind" not in row.keys() or row["adapter_kind"] is None
        else str(row["adapter_kind"])
    )
    funding_cost_bps = (
        None
        if "funding_cost_bps" not in row.keys() or row["funding_cost_bps"] is None
        else float(row["funding_cost_bps"])
    )
    borrow_fee_bps = (
        None
        if "borrow_fee_bps" not in row.keys() or row["borrow_fee_bps"] is None
        else float(row["borrow_fee_bps"])
    )
    roll_cost_bps = (
        None
        if "roll_cost_bps" not in row.keys() or row["roll_cost_bps"] is None
        else float(row["roll_cost_bps"])
    )
    financing_cost_bps = (
        None
        if "financing_cost_bps" not in row.keys() or row["financing_cost_bps"] is None
        else float(row["financing_cost_bps"])
    )
    contract_multiplier = (
        None
        if "contract_multiplier" not in row.keys() or row["contract_multiplier"] is None
        else float(row["contract_multiplier"])
    )
    contract_id = (
        None
        if "contract_id" not in row.keys() or row["contract_id"] is None
        else str(row["contract_id"])
    )
    contract_family = (
        None
        if "contract_family" not in row.keys() or row["contract_family"] is None
        else str(row["contract_family"])
    )
    quote_ccy = (
        None
        if "quote_ccy" not in row.keys() or row["quote_ccy"] is None
        else str(row["quote_ccy"])
    )
    collateral_ccy = (
        None
        if "collateral_ccy" not in row.keys() or row["collateral_ccy"] is None
        else str(row["collateral_ccy"])
    )
    roll_event = (
        None
        if "roll_event_json" not in row.keys() or row["roll_event_json"] is None
        else json.loads(str(row["roll_event_json"]))
    )
    if observation_spec_id is None and "signal_name" in row.keys() and row["signal_name"] is not None:
        observation_spec_id = str(row["signal_name"])
        observable_id = "daily_close"
        adapter_kind = "signal_noise_asset_observable"
    return EvaluationSnapshot(
        evaluation_id=str(row["evaluation_id"]),
        subject_id=(
            str(row["subject_id"])
            if "subject_id" in row.keys() and row["subject_id"] is not None
            else str(row["asset"])
        ),
        asset=str(row["asset"]),
        target_id=str(row["target_id"]),
        signal_id=str(row["signal_id"]),
        prediction_value=float(row["prediction_value"]),
        observation_value=float(row["observation_value"]),
        signed_edge=float(row["signed_edge"]),
        absolute_error=float(row["absolute_error"]),
        input_source=None if row["input_source"] is None else str(row["input_source"]),
        input_range_start=None
        if row["input_range_start"] is None
        else str(row["input_range_start"]),
        input_range_end=None
        if row["input_range_end"] is None
        else str(row["input_range_end"]),
        funding_cost_bps=funding_cost_bps,
        borrow_fee_bps=borrow_fee_bps,
        roll_cost_bps=roll_cost_bps,
        financing_cost_bps=financing_cost_bps,
        contract_multiplier=contract_multiplier,
        contract_id=contract_id,
        contract_family=contract_family,
        quote_ccy=quote_ccy,
        collateral_ccy=collateral_ccy,
        roll_event=roll_event,
        observation_spec_id=observation_spec_id,
        observable_id=observable_id,
        adapter_kind=adapter_kind,
        created_at=str(row["created_at"]),
    )


def _row_to_prediction(row: sqlite3.Row | None) -> PredictionRecord | None:
    if row is None:
        return None
    return PredictionRecord(
        evaluation_id=str(row["evaluation_id"]),
        signal_id=str(row["signal_id"]),
        subject_id=(
            str(row["subject_id"])
            if "subject_id" in row.keys() and row["subject_id"] is not None
            else str(row["asset"])
        ),
        asset=str(row["asset"]),
        target_id=str(row["target_id"]),
        value=float(row["value"]),
        recorded_at=str(row["recorded_at"]),
    )


def _row_to_observation(row: sqlite3.Row | None) -> ObservationRecord | None:
    if row is None:
        return None
    return ObservationRecord(
        evaluation_id=str(row["evaluation_id"]),
        subject_id=(
            str(row["subject_id"])
            if "subject_id" in row.keys() and row["subject_id"] is not None
            else str(row["asset"])
        ),
        asset=str(row["asset"]),
        target_id=str(row["target_id"]),
        value=float(row["value"]),
        recorded_at=str(row["recorded_at"]),
    )


def _row_to_signal_metric(
    row: sqlite3.Row | None,
) -> SignalMetricState | None:
    if row is None:
        return None
    return SignalMetricState(
        signal_id=str(row["signal_id"]),
        corr=float(row["corr"]),
        mmc=None if row["mmc"] is None else float(row["mmc"]),
        mmc_baseline_type=None
        if row["mmc_baseline_type"] is None
        else str(row["mmc_baseline_type"]),
        mmc_peer_count=int(row["mmc_peer_count"]),
        sample_count=int(row["sample_count"]),
        mmc_sample_count=int(row["mmc_sample_count"]),
        window_size=int(row["window_size"]),
        start_evaluation_id=None
        if row["start_evaluation_id"] is None
        else str(row["start_evaluation_id"]),
        end_evaluation_id=None if row["end_evaluation_id"] is None else str(row["end_evaluation_id"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_target(row: sqlite3.Row | None) -> TargetState | None:
    if row is None:
        return None
    return TargetState(
        target_id=str(row["target_id"]),
        definition_json=str(row["definition_json"]),
    )


def _row_to_observable(row: sqlite3.Row | None) -> ObservableState | None:
    if row is None:
        return None
    return ObservableState(
        observable_id=str(row["observable_id"]),
        definition_json=str(row["definition_json"]),
    )


def _row_to_observable(row: sqlite3.Row | None) -> ObservableState | None:
    if row is None:
        return None
    return ObservableState(
        observable_id=str(row["observable_id"]),
        definition_json=str(row["definition_json"]),
    )


def _row_to_subject_set(row: sqlite3.Row | None) -> SubjectSetState | None:
    if row is None:
        return None
    return SubjectSetState(
        subject_set_id=str(row["subject_set_id"]),
        definition_json=str(row["definition_json"]),
    )


def _row_to_meta_prediction(row: sqlite3.Row | None) -> MetaPredictionState | None:
    if row is None:
        return None
    return MetaPredictionState(
        evaluation_id=str(row["evaluation_id"]),
        subject_id=(
            str(row["subject_id"])
            if "subject_id" in row.keys() and row["subject_id"] is not None
            else str(row["asset"])
        ),
        asset=str(row["asset"]),
        target_id=str(row["target_id"]),
        aggregation_kind=str(row["aggregation_kind"]),
        value=float(row["value"]),
        contributor_count=int(row["contributor_count"]),
        details_json=None if row["details_json"] is None else str(row["details_json"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_meta_prediction_metric(row: sqlite3.Row | None) -> MetaPredictionMetricState | None:
    if row is None:
        return None
    return MetaPredictionMetricState(
        aggregation_kind=str(row["aggregation_kind"]),
        subject_id=(
            str(row["subject_id"])
            if "subject_id" in row.keys() and row["subject_id"] is not None
            else str(row["asset"])
        ),
        asset=str(row["asset"]),
        target_id=str(row["target_id"]),
        corr=float(row["corr"]),
        sample_count=int(row["sample_count"]),
        window_size=int(row["window_size"]),
        start_evaluation_id=None
        if row["start_evaluation_id"] is None
        else str(row["start_evaluation_id"]),
        end_evaluation_id=None
        if row["end_evaluation_id"] is None
        else str(row["end_evaluation_id"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_portfolio_decision(row: sqlite3.Row | None) -> PortfolioDecisionState | None:
    if row is None:
        return None
    return PortfolioDecisionState(
        portfolio_id=str(row["portfolio_id"]),
        subject_id=str(row["subject_id"]),
        target_id=str(row["target_id"]),
        aggregation_kind=str(row["aggregation_kind"]),
        as_of=str(row["as_of"]),
        target_weight=float(row["target_weight"]),
        position_delta=float(row["position_delta"]),
        target_notional=None
        if row["target_notional"] is None
        else float(row["target_notional"]),
        target_quantity=None
        if row["target_quantity"] is None
        else float(row["target_quantity"]),
        entry_allowed=bool(int(row["entry_allowed"])),
        risk_scale=float(row["risk_scale"]),
        details_json=None if row["details_json"] is None else str(row["details_json"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_evaluation_decision_trace_step(
    row: sqlite3.Row | None,
) -> EvaluationDecisionTraceStepState | None:
    if row is None:
        return None
    return EvaluationDecisionTraceStepState(
        evaluation_report_id=str(row["evaluation_report_id"]),
        evaluation_task_id=str(row["evaluation_task_id"]),
        evaluation_fold_label=str(row["evaluation_fold_label"]),
        evaluation_range_label=str(row["evaluation_range_label"]),
        variant=str(row["variant"]),
        step_index=int(row["step_index"]),
        step_as_of=str(row["step_as_of"]),
        step_granularity=str(row["step_granularity"]),
        target_id=str(row["target_id"]),
        subject_set_id=str(row["subject_set_id"]),
        gross_return=float(row["gross_return"]),
        net_return=float(row["net_return"]),
        gross_pnl_notional=float(row["gross_pnl_notional"]),
        net_pnl_notional=float(row["net_pnl_notional"]),
        turnover=float(row["turnover"]),
        traded_notional=float(row["traded_notional"]),
        cost_notional=float(row["cost_notional"]),
        funding_cost_notional=float(row["funding_cost_notional"]),
        borrow_cost_notional=float(row["borrow_cost_notional"]),
        roll_cost_notional=float(row["roll_cost_notional"]),
        gross_leverage_exposure=float(row["gross_leverage_exposure"]),
        net_leverage_exposure=float(row["net_leverage_exposure"]),
        long_leverage_exposure=float(row["long_leverage_exposure"]),
        short_leverage_exposure=float(row["short_leverage_exposure"]),
        gross_notional_exposure=float(row["gross_notional_exposure"]),
        net_notional_exposure=float(row["net_notional_exposure"]),
        long_notional_exposure=float(row["long_notional_exposure"]),
        short_notional_exposure=float(row["short_notional_exposure"]),
        gross_equity=float(row["gross_equity"]),
        net_equity=float(row["net_equity"]),
        created_at=str(row["created_at"]),
    )


def _row_to_evaluation_decision_trace_subject_step(
    row: sqlite3.Row | None,
) -> EvaluationDecisionTraceSubjectStepState | None:
    if row is None:
        return None
    return EvaluationDecisionTraceSubjectStepState(
        evaluation_report_id=str(row["evaluation_report_id"]),
        evaluation_task_id=str(row["evaluation_task_id"]),
        evaluation_fold_label=str(row["evaluation_fold_label"]),
        evaluation_range_label=str(row["evaluation_range_label"]),
        variant=str(row["variant"]),
        step_index=int(row["step_index"]),
        subject_id=str(row["subject_id"]),
        asset_class=None if row["asset_class"] is None else str(row["asset_class"]),
        cluster=None if row["cluster"] is None else str(row["cluster"]),
        signal_value=float(row["signal_value"]),
        realized_return=float(row["realized_return"]),
        target_weight=float(row["target_weight"]),
        position_delta=float(row["position_delta"]),
        target_notional=float(row["target_notional"]),
        traded_notional=float(row["traded_notional"]),
        gross_pnl_notional=float(row["gross_pnl_notional"]),
        execution_cost_notional=float(row["execution_cost_notional"]),
        funding_cost_notional=float(row["funding_cost_notional"]),
        borrow_cost_notional=float(row["borrow_cost_notional"]),
        roll_cost_notional=float(row["roll_cost_notional"]),
        cost_notional=float(row["cost_notional"]),
        net_pnl_notional=float(row["net_pnl_notional"]),
        net_return_contribution=float(row["net_return_contribution"]),
        risk_scale=float(row["risk_scale"]),
        entry_allowed=bool(int(row["entry_allowed"])),
        funding_cost_bps=float(row["funding_cost_bps"]),
        borrow_fee_bps=float(row["borrow_fee_bps"]),
        roll_cost_bps=float(row["roll_cost_bps"]),
        contract_multiplier=(
            None if row["contract_multiplier"] is None else float(row["contract_multiplier"])
        ),
        target_contracts=(
            None if row["target_contracts"] is None else float(row["target_contracts"])
        ),
        traded_contracts=(
            None if row["traded_contracts"] is None else float(row["traded_contracts"])
        ),
        created_at=str(row["created_at"]),
    )


def _row_to_validation_run(row: sqlite3.Row | None) -> ValidationRunState | None:
    if row is None:
        return None
    return ValidationRunState(
        run_id=str(row["run_id"]),
        spec_json=str(row["spec_json"]),
        created_at=str(row["created_at"]),
        criteria_json=(
            None
            if "criteria_json" not in row.keys() or row["criteria_json"] is None
            else str(row["criteria_json"])
        ),
        summary_json=(
            None
            if "summary_json" not in row.keys() or row["summary_json"] is None
            else str(row["summary_json"])
        ),
    )


def _row_to_evaluation_report(row: sqlite3.Row | None) -> EvaluationReportState | None:
    if row is None:
        return None
    return EvaluationReportState(
        evaluation_report_id=str(row["evaluation_report_id"]),
        evaluation_spec_id=str(row["evaluation_spec_id"]),
        report_json=str(row["report_json"]),
        created_at=str(row["created_at"]),
    )


def _row_to_strategy_adaptation_state(
    row: sqlite3.Row | None,
) -> StrategyAdaptationStateRecord | None:
    if row is None:
        return None
    return StrategyAdaptationStateRecord(
        strategy_id=str(row["strategy_id"]),
        signal_discovery_id=(
            None
            if row["signal_discovery_id"] is None or str(row["signal_discovery_id"]) == ""
            else str(row["signal_discovery_id"])
        ),
        state_json=str(row["state_json"]),
        created_at=str(row["created_at"]),
    )


def _row_to_screening_result(row: sqlite3.Row | None) -> ScreeningResultState | None:
    if row is None:
        return None
    return ScreeningResultState(
        screening_result_id=str(row["screening_result_id"]),
        signal_discovery_id=str(row["signal_discovery_id"]),
        result_json=str(row["result_json"]),
        created_at=str(row["created_at"]),
    )


def _row_to_compressed_belief(row: sqlite3.Row | None) -> CompressedBeliefState | None:
    if row is None:
        return None
    return CompressedBeliefState(
        compressed_belief_id=str(row["compressed_belief_id"]),
        signal_discovery_id=str(row["signal_discovery_id"]),
        belief_json=str(row["belief_json"]),
        created_at=str(row["created_at"]),
    )


def _row_to_signal_discovery_run(
    row: sqlite3.Row | None,
) -> SignalDiscoveryRunState | None:
    if row is None:
        return None
    return SignalDiscoveryRunState(
        signal_discovery_run_id=str(row["signal_discovery_run_id"]),
        signal_discovery_id=str(row["signal_discovery_id"]),
        run_json=str(row["run_json"]),
        created_at=str(row["created_at"]),
    )


def _row_to_initial_strategy_state(
    row: sqlite3.Row | None,
) -> InitialStrategyStateRecord | None:
    if row is None:
        return None
    discovery_value = (
        None if row["signal_discovery_id"] is None else str(row["signal_discovery_id"])
    )
    signal_train_value = row["signal_train_id"]
    if signal_train_value is None:
        signal_train_id = (
            f"signal-train:{discovery_value}"
            if discovery_value is not None
            else ""
        )
    else:
        signal_train_id = str(signal_train_value)
    return InitialStrategyStateRecord(
        initial_strategy_state_id=str(row["initial_strategy_state_id"]),
        strategy_id=str(row["strategy_id"]),
        signal_train_id=signal_train_id,
        signal_discovery_id=discovery_value,
        artifact_json=str(row["artifact_json"]),
        created_at=str(row["created_at"]),
    )


def _row_to_trading_strategy(row: sqlite3.Row | None) -> TradingStrategyState | None:
    if row is None:
        return None
    return TradingStrategyState(
        strategy_id=str(row["strategy_id"]),
        spec_json=str(row["spec_json"]),
        created_at=str(row["created_at"]),
    )


def _row_to_evaluation_task(row: sqlite3.Row | None) -> EvaluationTaskState | None:
    if row is None:
        return None
    return EvaluationTaskState(
        evaluation_task_id=str(row["evaluation_task_id"]),
        task_json=str(row["task_json"]),
        created_at=str(row["created_at"]),
    )


def _row_to_evaluation_job_spec(
    row: sqlite3.Row | None,
) -> EvaluationJobSpecState | None:
    if row is None:
        return None
    return EvaluationJobSpecState(
        evaluation_task_id=str(row["evaluation_task_id"]),
        job_spec_json=str(row["job_spec_json"]),
        created_at=str(row["created_at"]),
    )


def _row_to_validation_signal_result(
    row: sqlite3.Row | None,
) -> ValidationSignalResultState | None:
    if row is None:
        return None
    return ValidationSignalResultState(
        run_id=str(row["run_id"]),
        date_range_label=str(row["date_range_label"]),
        start_date=str(row["start_date"]),
        end_date=str(row["end_date"]),
        target_id=str(row["target_id"]),
        signal_id=str(row["signal_id"]),
        window_size=int(row["window_size"]),
        corr=float(row["corr"]),
        mmc=None if row["mmc"] is None else float(row["mmc"]),
        sample_count=int(row["sample_count"]),
        mmc_sample_count=int(row["mmc_sample_count"]),
        mmc_peer_count=int(row["mmc_peer_count"]),
        mmc_baseline_type=None
        if row["mmc_baseline_type"] is None
        else str(row["mmc_baseline_type"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_signal_metric_legacy(
    row: sqlite3.Row | None,
) -> SignalMetricState | None:
    return _row_to_signal_metric(row)


def _row_to_validation_signal_result_legacy(
    row: sqlite3.Row | None,
) -> ValidationSignalResultState | None:
    return _row_to_validation_signal_result(row)


def _row_to_validation_meta_result(
    row: sqlite3.Row | None,
) -> ValidationMetaResultState | None:
    if row is None:
        return None
    return ValidationMetaResultState(
        run_id=str(row["run_id"]),
        date_range_label=str(row["date_range_label"]),
        start_date=str(row["start_date"]),
        end_date=str(row["end_date"]),
        target_id=str(row["target_id"]),
        aggregation_kind=str(row["aggregation_kind"]),
        window_size=int(row["window_size"]),
        corr=float(row["corr"]),
        sample_count=int(row["sample_count"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_validation_decision_result(
    row: sqlite3.Row | None,
) -> ValidationDecisionResultState | None:
    if row is None:
        return None
    return ValidationDecisionResultState(
        run_id=str(row["run_id"]),
        date_range_label=str(row["date_range_label"]),
        start_date=str(row["start_date"]),
        end_date=str(row["end_date"]),
        target_id=str(row["target_id"]),
        subject_set_id=str(row["subject_set_id"]),
        aggregation_kind=str(row["aggregation_kind"]),
        window_size=int(row["window_size"]),
        gross_return_total=float(row["gross_return_total"]),
        net_return_total=float(row["net_return_total"]),
        max_drawdown=float(row["max_drawdown"]),
        mean_turnover=float(row["mean_turnover"]),
        mean_gross_notional_exposure=float(row["mean_gross_notional_exposure"]),
        mean_net_notional_exposure=float(row["mean_net_notional_exposure"]),
        mean_long_notional_exposure=float(row["mean_long_notional_exposure"]),
        mean_short_notional_exposure=float(row["mean_short_notional_exposure"]),
        mean_traded_notional=float(row["mean_traded_notional"]),
        cost_notional_total=float(row["cost_notional_total"]),
        funding_cost_notional_total=float(row["funding_cost_notional_total"]),
        borrow_cost_notional_total=float(row["borrow_cost_notional_total"]),
        roll_cost_notional_total=float(row["roll_cost_notional_total"]),
        step_count=int(row["step_count"]),
        updated_at=str(row["updated_at"]),
    )


class EvaluationStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS targets (
                target_id TEXT PRIMARY KEY,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observables (
                observable_id TEXT PRIMARY KEY,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS subject_sets (
                subject_set_id TEXT PRIMARY KEY,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signal_discoveries (
                signal_discovery_id TEXT PRIMARY KEY,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_specs (
                evaluation_spec_id TEXT PRIMARY KEY,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signal_specs (
                signal_id TEXT PRIMARY KEY,
                target_id TEXT NOT NULL,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                specification_signal_id TEXT,
                subject_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                definition_json TEXT,
                status TEXT NOT NULL,
                prediction_count INTEGER NOT NULL,
                observation_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                evaluation_id TEXT NOT NULL,
                signal_id TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                value REAL NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (evaluation_id, signal_id)
            );

            CREATE TABLE IF NOT EXISTS observations (
                evaluation_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                value REAL NOT NULL,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signal_metrics (
                signal_id TEXT PRIMARY KEY,
                corr REAL NOT NULL,
                mmc REAL,
                mmc_baseline_type TEXT,
                mmc_peer_count INTEGER NOT NULL,
                sample_count INTEGER NOT NULL,
                mmc_sample_count INTEGER NOT NULL,
                window_size INTEGER NOT NULL,
                start_evaluation_id TEXT,
                end_evaluation_id TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observation_frame_cache (
                cache_key TEXT PRIMARY KEY,
                frame_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_snapshots (
                evaluation_id TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                signal_id TEXT NOT NULL,
                prediction_value REAL NOT NULL,
                observation_value REAL NOT NULL,
                signed_edge REAL NOT NULL,
                absolute_error REAL NOT NULL,
                input_source TEXT,
                input_range_start TEXT,
                input_range_end TEXT,
                funding_cost_bps REAL,
                borrow_fee_bps REAL,
                roll_cost_bps REAL,
                financing_cost_bps REAL,
                contract_multiplier REAL,
                contract_id TEXT,
                contract_family TEXT,
                quote_ccy TEXT,
                collateral_ccy TEXT,
                roll_event_json TEXT,
                observation_spec_id TEXT,
                observable_id TEXT,
                adapter_kind TEXT,
                signal_name TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (evaluation_id, signal_id)
            );

            CREATE TABLE IF NOT EXISTS meta_predictions (
                evaluation_id TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                aggregation_kind TEXT NOT NULL,
                value REAL NOT NULL,
                contributor_count INTEGER NOT NULL,
                details_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (evaluation_id, aggregation_kind)
            );

            CREATE TABLE IF NOT EXISTS meta_prediction_metrics (
                aggregation_kind TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                corr REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                window_size INTEGER NOT NULL,
                start_evaluation_id TEXT,
                end_evaluation_id TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (aggregation_kind, subject_id, target_id)
            );

            CREATE TABLE IF NOT EXISTS validation_runs (
                run_id TEXT PRIMARY KEY,
                spec_json TEXT NOT NULL,
                criteria_json TEXT,
                summary_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS screening_results (
                screening_result_id TEXT PRIMARY KEY,
                signal_discovery_id TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS compressed_beliefs (
                compressed_belief_id TEXT PRIMARY KEY,
                signal_discovery_id TEXT NOT NULL,
                belief_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signal_discovery_runs (
                signal_discovery_run_id TEXT PRIMARY KEY,
                signal_discovery_id TEXT NOT NULL,
                run_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS initial_strategy_states (
                initial_strategy_state_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                signal_train_id TEXT NOT NULL DEFAULT '',
                signal_discovery_id TEXT,
                artifact_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS strategy_specs (
                strategy_id TEXT PRIMARY KEY,
                spec_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_tasks (
                evaluation_task_id TEXT PRIMARY KEY,
                task_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_job_specs (
                evaluation_task_id TEXT PRIMARY KEY,
                job_spec_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signal_discovery_run_evaluation_snapshots (
                signal_discovery_run_id TEXT NOT NULL,
                evaluation_id TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                signal_id TEXT NOT NULL,
                prediction_value REAL NOT NULL,
                observation_value REAL NOT NULL,
                signed_edge REAL NOT NULL,
                absolute_error REAL NOT NULL,
                input_source TEXT,
                input_range_start TEXT,
                input_range_end TEXT,
                funding_cost_bps REAL,
                borrow_fee_bps REAL,
                roll_cost_bps REAL,
                financing_cost_bps REAL,
                contract_multiplier REAL,
                contract_id TEXT,
                contract_family TEXT,
                quote_ccy TEXT,
                collateral_ccy TEXT,
                roll_event_json TEXT,
                observation_spec_id TEXT,
                observable_id TEXT,
                adapter_kind TEXT,
                signal_name TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (signal_discovery_run_id, evaluation_id, signal_id)
            );

            CREATE TABLE IF NOT EXISTS evaluation_reports (
                evaluation_report_id TEXT PRIMARY KEY,
                evaluation_spec_id TEXT NOT NULL,
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_decision_trace_steps (
                evaluation_report_id TEXT NOT NULL,
                evaluation_task_id TEXT NOT NULL,
                evaluation_fold_label TEXT NOT NULL,
                evaluation_range_label TEXT NOT NULL,
                variant TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                step_as_of TEXT NOT NULL,
                step_granularity TEXT NOT NULL,
                target_id TEXT NOT NULL,
                subject_set_id TEXT NOT NULL DEFAULT '',
                gross_return REAL NOT NULL,
                net_return REAL NOT NULL,
                gross_pnl_notional REAL NOT NULL,
                net_pnl_notional REAL NOT NULL,
                turnover REAL NOT NULL,
                traded_notional REAL NOT NULL,
                cost_notional REAL NOT NULL,
                funding_cost_notional REAL NOT NULL,
                borrow_cost_notional REAL NOT NULL,
                roll_cost_notional REAL NOT NULL,
                gross_leverage_exposure REAL NOT NULL,
                net_leverage_exposure REAL NOT NULL,
                long_leverage_exposure REAL NOT NULL,
                short_leverage_exposure REAL NOT NULL,
                gross_notional_exposure REAL NOT NULL,
                net_notional_exposure REAL NOT NULL,
                long_notional_exposure REAL NOT NULL,
                short_notional_exposure REAL NOT NULL,
                gross_equity REAL NOT NULL,
                net_equity REAL NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (
                    evaluation_report_id, evaluation_task_id, evaluation_fold_label,
                    evaluation_range_label, variant, step_index
                )
            );

            CREATE TABLE IF NOT EXISTS evaluation_decision_trace_subject_steps (
                evaluation_report_id TEXT NOT NULL,
                evaluation_task_id TEXT NOT NULL,
                evaluation_fold_label TEXT NOT NULL,
                evaluation_range_label TEXT NOT NULL,
                variant TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                subject_id TEXT NOT NULL,
                asset_class TEXT,
                cluster TEXT,
                signal_value REAL NOT NULL,
                realized_return REAL NOT NULL,
                target_weight REAL NOT NULL,
                position_delta REAL NOT NULL,
                target_notional REAL NOT NULL,
                traded_notional REAL NOT NULL,
                gross_pnl_notional REAL NOT NULL,
                execution_cost_notional REAL NOT NULL,
                funding_cost_notional REAL NOT NULL,
                borrow_cost_notional REAL NOT NULL,
                roll_cost_notional REAL NOT NULL,
                cost_notional REAL NOT NULL,
                net_pnl_notional REAL NOT NULL,
                net_return_contribution REAL NOT NULL,
                risk_scale REAL NOT NULL,
                entry_allowed INTEGER NOT NULL,
                funding_cost_bps REAL NOT NULL,
                borrow_fee_bps REAL NOT NULL,
                roll_cost_bps REAL NOT NULL,
                contract_multiplier REAL,
                target_contracts REAL,
                traded_contracts REAL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (
                    evaluation_report_id, evaluation_task_id, evaluation_fold_label,
                    evaluation_range_label, variant, step_index, subject_id
                )
            );

            CREATE TABLE IF NOT EXISTS strategy_adaptation_states (
                strategy_id TEXT PRIMARY KEY,
                signal_discovery_id TEXT,
                signal_train_id TEXT,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS validation_signal_results (
                run_id TEXT NOT NULL,
                date_range_label TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                target_id TEXT NOT NULL,
                signal_id TEXT NOT NULL,
                window_size INTEGER NOT NULL,
                corr REAL NOT NULL,
                mmc REAL,
                sample_count INTEGER NOT NULL,
                mmc_sample_count INTEGER NOT NULL,
                mmc_peer_count INTEGER NOT NULL,
                mmc_baseline_type TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, date_range_label, target_id, signal_id, window_size)
            );

            CREATE TABLE IF NOT EXISTS validation_meta_results (
                run_id TEXT NOT NULL,
                date_range_label TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                target_id TEXT NOT NULL,
                aggregation_kind TEXT NOT NULL,
                window_size INTEGER NOT NULL,
                corr REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, date_range_label, target_id, aggregation_kind, window_size)
            );

            CREATE TABLE IF NOT EXISTS validation_decision_results (
                run_id TEXT NOT NULL,
                date_range_label TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                target_id TEXT NOT NULL,
                subject_set_id TEXT NOT NULL DEFAULT '',
                aggregation_kind TEXT NOT NULL,
                window_size INTEGER NOT NULL,
                gross_return_total REAL NOT NULL,
                net_return_total REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                mean_turnover REAL NOT NULL,
                mean_gross_notional_exposure REAL NOT NULL DEFAULT 0.0,
                mean_net_notional_exposure REAL NOT NULL DEFAULT 0.0,
                mean_long_notional_exposure REAL NOT NULL DEFAULT 0.0,
                mean_short_notional_exposure REAL NOT NULL DEFAULT 0.0,
                mean_traded_notional REAL NOT NULL DEFAULT 0.0,
                cost_notional_total REAL NOT NULL DEFAULT 0.0,
                funding_cost_notional_total REAL NOT NULL DEFAULT 0.0,
                borrow_cost_notional_total REAL NOT NULL DEFAULT 0.0,
                roll_cost_notional_total REAL NOT NULL DEFAULT 0.0,
                step_count INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (
                    run_id, date_range_label, target_id, subject_set_id,
                    aggregation_kind, window_size
                )
            );

            CREATE TABLE IF NOT EXISTS portfolio_decisions (
                portfolio_id TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                aggregation_kind TEXT NOT NULL,
                as_of TEXT NOT NULL,
                target_weight REAL NOT NULL,
                position_delta REAL NOT NULL,
                target_notional REAL,
                target_quantity REAL,
                entry_allowed INTEGER NOT NULL,
                risk_scale REAL NOT NULL,
                details_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (portfolio_id, subject_id, target_id, aggregation_kind, as_of)
            );
            """
        )
        self._ensure_strategy_adaptation_state_schema()
        self._seed_builtin_targets()
        self._seed_builtin_observables()
        self._ensure_subject_first_runtime_schema()
        self._ensure_signal_discovery_run_snapshot_schema()
        self._ensure_signal_spec_schema()
        self._ensure_validation_run_columns()
        self._ensure_validation_decision_result_columns()
        self.conn.commit()

    def _ensure_strategy_adaptation_state_schema(self) -> None:
        legacy_table_exists = self.conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'online_learning_states'
            """
        ).fetchone()
        if legacy_table_exists is not None:
            legacy_columns = self.conn.execute(
                "PRAGMA table_info(online_learning_states)"
            ).fetchall()
            legacy_column_names = {str(row["name"]) for row in legacy_columns}
            resolved_strategy_id = (
                "CASE WHEN strategy_id IS NULL OR strategy_id = '' "
                "THEN signal_discovery_id ELSE strategy_id END"
                if "strategy_id" in legacy_column_names
                else "signal_discovery_id"
            )
            resolved_signal_train_id = (
                "CASE WHEN signal_train_id IS NULL OR signal_train_id = '' "
                "THEN 'signal-train:' || signal_discovery_id ELSE signal_train_id END"
                if "signal_train_id" in legacy_column_names
                else "'signal-train:' || signal_discovery_id"
            )
            with self.conn:
                self.conn.execute(
                    f"""
                    INSERT OR REPLACE INTO strategy_adaptation_states (
                        strategy_id,
                        signal_discovery_id,
                        signal_train_id,
                        state_json,
                        created_at
                    )
                    SELECT
                        {resolved_strategy_id},
                        NULLIF(signal_discovery_id, ''),
                        {resolved_signal_train_id},
                        state_json,
                        created_at
                    FROM online_learning_states
                    WHERE {resolved_strategy_id} IS NOT NULL
                      AND {resolved_strategy_id} != ''
                    """
                )
                self.conn.execute("DROP TABLE online_learning_states")
        columns = self.conn.execute(
            "PRAGMA table_info(strategy_adaptation_states)"
        ).fetchall()
        if not columns:
            return
        column_names = {str(row["name"]) for row in columns}
        primary_key_columns = [
            str(row["name"])
            for row in columns
            if int(row["pk"]) > 0
        ]
        requires_rebuild = primary_key_columns != ["strategy_id"]
        if not requires_rebuild:
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_strategy_adaptation_states_signal_discovery_id
                ON strategy_adaptation_states(signal_discovery_id)
                """
            )
            return
        resolved_strategy_id = (
            "CASE WHEN strategy_id IS NULL OR strategy_id = '' "
            "THEN signal_discovery_id ELSE strategy_id END"
            if "strategy_id" in column_names
            else "signal_discovery_id"
        )
        resolved_signal_train_id = (
            "CASE WHEN signal_train_id IS NULL OR signal_train_id = '' "
            "THEN 'signal-train:' || signal_discovery_id ELSE signal_train_id END"
            if "signal_train_id" in column_names
            else "'signal-train:' || signal_discovery_id"
        )
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE strategy_adaptation_states_v2 (
                    strategy_id TEXT PRIMARY KEY,
                    signal_discovery_id TEXT,
                    signal_train_id TEXT,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                f"""
                INSERT OR REPLACE INTO strategy_adaptation_states_v2 (
                    strategy_id,
                    signal_discovery_id,
                    signal_train_id,
                    state_json,
                    created_at
                )
                SELECT
                    {resolved_strategy_id},
                    NULLIF(signal_discovery_id, ''),
                    {resolved_signal_train_id},
                    state_json,
                    created_at
                FROM strategy_adaptation_states
                WHERE {resolved_strategy_id} IS NOT NULL
                  AND {resolved_strategy_id} != ''
                """
            )
            self.conn.execute("DROP TABLE strategy_adaptation_states")
            self.conn.execute(
                "ALTER TABLE strategy_adaptation_states_v2 RENAME TO strategy_adaptation_states"
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_strategy_adaptation_states_signal_discovery_id
                ON strategy_adaptation_states(signal_discovery_id)
                """
            )

    def _ensure_subject_first_runtime_schema(self) -> None:
        table_columns = {
            table: {
                str(row["name"])
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for table in (
                "signals",
                "predictions",
                "observations",
                "evaluation_snapshots",
                "meta_predictions",
                "meta_prediction_metrics",
                "initial_strategy_states",
            )
        }
        required_columns = {
            "signals.signal_spec_id": "TEXT",
            "signals": "TEXT NOT NULL DEFAULT ''",
            "predictions": "TEXT NOT NULL DEFAULT ''",
            "observations": "TEXT NOT NULL DEFAULT ''",
            "evaluation_snapshots": "TEXT NOT NULL DEFAULT ''",
            "evaluation_snapshots.observation_spec_id": "TEXT",
            "evaluation_snapshots.observable_id": "TEXT",
            "evaluation_snapshots.adapter_kind": "TEXT",
            "evaluation_snapshots.funding_cost_bps": "REAL",
            "evaluation_snapshots.borrow_fee_bps": "REAL",
            "evaluation_snapshots.roll_cost_bps": "REAL",
            "evaluation_snapshots.financing_cost_bps": "REAL",
            "evaluation_snapshots.contract_multiplier": "REAL",
            "evaluation_snapshots.contract_id": "TEXT",
            "evaluation_snapshots.contract_family": "TEXT",
            "evaluation_snapshots.quote_ccy": "TEXT",
            "evaluation_snapshots.collateral_ccy": "TEXT",
            "evaluation_snapshots.roll_event_json": "TEXT",
            "meta_predictions": "TEXT NOT NULL DEFAULT ''",
            "initial_strategy_states.strategy_id": "TEXT NOT NULL DEFAULT ''",
            "initial_strategy_states.signal_train_id": "TEXT NOT NULL DEFAULT ''",
        }
        for key, definition in required_columns.items():
            if "." in key:
                table_name, column_name = key.split(".", 1)
                if column_name in table_columns[table_name]:
                    continue
                self.conn.execute(
                    f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN {column_name} {definition}
                    """
                )
                continue
            table_name = key
            if "subject_id" in table_columns[table_name]:
                continue
            self.conn.execute(
                f"""
                ALTER TABLE {table_name}
                ADD COLUMN subject_id {definition}
                """
            )
        self.conn.execute(
            """
            UPDATE initial_strategy_states
            SET strategy_id = signal_discovery_id
            WHERE strategy_id = ''
            """
        )
        self.conn.execute(
            """
            UPDATE initial_strategy_states
            SET signal_train_id = 'signal-train:' || signal_discovery_id
            WHERE signal_train_id = ''
              AND signal_discovery_id IS NOT NULL
            """
        )
        self._backfill_runtime_subject_ids()
        self._ensure_subject_first_meta_prediction_metrics_table()

    def _ensure_signal_spec_schema(self) -> None:
        self._seed_builtin_signal_specs()

    def _backfill_runtime_subject_ids(self) -> None:
        signal_rows = self.conn.execute(
            """
            SELECT signal_id, asset
            FROM signals
            WHERE subject_id = ''
            """
        ).fetchall()
        for row in signal_rows:
            self.conn.execute(
                """
                UPDATE signals
                SET subject_id = ?
                WHERE signal_id = ?
                """,
                (
                    subject_id_for_signal(
                        signal_id=str(row["signal_id"]),
                        asset=str(row["asset"]),
                    ),
                    str(row["signal_id"]),
                ),
            )
        self.conn.execute(
            """
            UPDATE predictions
            SET subject_id = COALESCE(
                (
                    SELECT h.subject_id
                    FROM signals AS h
                    WHERE h.signal_id = predictions.signal_id
                ),
                asset
            )
            WHERE subject_id = ''
            """
        )
        self.conn.execute(
            """
            UPDATE observations
            SET subject_id = COALESCE(
                (
                    SELECT p.subject_id
                    FROM predictions AS p
                    WHERE p.evaluation_id = observations.evaluation_id
                    ORDER BY p.signal_id ASC
                    LIMIT 1
                ),
                asset
            )
            WHERE subject_id = ''
            """
        )
        self.conn.execute(
            """
            UPDATE evaluation_snapshots
            SET subject_id = COALESCE(
                (
                    SELECT h.subject_id
                    FROM signals AS h
                    WHERE h.signal_id = evaluation_snapshots.signal_id
                ),
                asset
            )
            WHERE subject_id = ''
            """
        )

    def _ensure_signal_discovery_run_snapshot_schema(self) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute(
                "PRAGMA table_info(signal_discovery_run_evaluation_snapshots)"
            ).fetchall()
        }
        required_columns = {
            "subject_id": "TEXT NOT NULL DEFAULT ''",
            "observation_spec_id": "TEXT",
            "observable_id": "TEXT",
            "adapter_kind": "TEXT",
            "signal_name": "TEXT",
            "funding_cost_bps": "REAL",
            "borrow_fee_bps": "REAL",
            "roll_cost_bps": "REAL",
            "financing_cost_bps": "REAL",
            "contract_multiplier": "REAL",
            "contract_id": "TEXT",
            "contract_family": "TEXT",
            "quote_ccy": "TEXT",
            "collateral_ccy": "TEXT",
            "roll_event_json": "TEXT",
        }
        for column_name, definition in required_columns.items():
            if column_name in columns:
                continue
            self.conn.execute(
                f"""
                ALTER TABLE signal_discovery_run_evaluation_snapshots
                ADD COLUMN {column_name} {definition}
                """
            )
        self.conn.execute(
            """
            UPDATE meta_predictions
            SET subject_id = COALESCE(
                (
                    SELECT o.subject_id
                    FROM observations AS o
                    WHERE o.evaluation_id = meta_predictions.evaluation_id
                ),
                asset
            )
            WHERE subject_id = ''
            """
        )

    def _ensure_subject_first_meta_prediction_metrics_table(self) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute(
                "PRAGMA table_info(meta_prediction_metrics)"
            ).fetchall()
        }
        if "subject_id" in columns and any(
            int(row["pk"]) > 0 and str(row["name"]) == "subject_id"
            for row in self.conn.execute(
                "PRAGMA table_info(meta_prediction_metrics)"
            ).fetchall()
        ):
            return
        rows = self.conn.execute(
            """
            SELECT aggregation_kind, asset, target_id, corr, sample_count,
                   window_size, start_evaluation_id, end_evaluation_id, updated_at
            FROM meta_prediction_metrics
            """
        ).fetchall()
        self.conn.executescript(
            """
            DROP TABLE meta_prediction_metrics;
            CREATE TABLE meta_prediction_metrics (
                aggregation_kind TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                target_id TEXT NOT NULL,
                corr REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                window_size INTEGER NOT NULL,
                start_evaluation_id TEXT,
                end_evaluation_id TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (aggregation_kind, subject_id, target_id)
            );
            """
        )
        if not rows:
            return
        self.conn.executemany(
            """
            INSERT INTO meta_prediction_metrics (
                aggregation_kind, subject_id, asset, target_id, corr,
                sample_count, window_size, start_evaluation_id,
                end_evaluation_id, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row["aggregation_kind"]),
                    str(row["asset"]),
                    str(row["asset"]),
                    str(row["target_id"]),
                    float(row["corr"]),
                    int(row["sample_count"]),
                    int(row["window_size"]),
                    None
                    if row["start_evaluation_id"] is None
                    else str(row["start_evaluation_id"]),
                    None
                    if row["end_evaluation_id"] is None
                    else str(row["end_evaluation_id"]),
                    str(row["updated_at"]),
                )
                for row in rows
            ],
        )

    def _ensure_validation_decision_result_columns(self) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute(
                "PRAGMA table_info(validation_decision_results)"
            ).fetchall()
        }
        required_columns = {
            "subject_set_id": "TEXT NOT NULL DEFAULT ''",
            "mean_gross_notional_exposure": "REAL NOT NULL DEFAULT 0.0",
            "mean_net_notional_exposure": "REAL NOT NULL DEFAULT 0.0",
            "mean_long_notional_exposure": "REAL NOT NULL DEFAULT 0.0",
            "mean_short_notional_exposure": "REAL NOT NULL DEFAULT 0.0",
            "mean_traded_notional": "REAL NOT NULL DEFAULT 0.0",
            "cost_notional_total": "REAL NOT NULL DEFAULT 0.0",
            "funding_cost_notional_total": "REAL NOT NULL DEFAULT 0.0",
            "borrow_cost_notional_total": "REAL NOT NULL DEFAULT 0.0",
            "roll_cost_notional_total": "REAL NOT NULL DEFAULT 0.0",
        }
        for name, definition in required_columns.items():
            if name in columns:
                continue
            self.conn.execute(
                f"""
                ALTER TABLE validation_decision_results
                ADD COLUMN {name} {definition}
                """
            )
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_validation_decision_results_scope
            ON validation_decision_results (
                run_id, date_range_label, target_id, subject_set_id,
                aggregation_kind, window_size
            )
            """
        )

    def _ensure_validation_run_columns(self) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute(
                "PRAGMA table_info(validation_runs)"
            ).fetchall()
        }
        required_columns = {
            "criteria_json": "TEXT",
            "summary_json": "TEXT",
        }
        for name, definition in required_columns.items():
            if name in columns:
                continue
            self.conn.execute(
                f"""
                ALTER TABLE validation_runs
                ADD COLUMN {name} {definition}
                """
            )

    def _seed_builtin_targets(self) -> None:
        timestamp = _utc_now()
        for definition in list_target_definitions():
            self.conn.execute(
                """
                INSERT INTO targets (target_id, definition_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(target_id) DO NOTHING
                """,
                (
                    definition.target_id,
                    json.dumps(definition.to_document(), sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )

    def _seed_builtin_observables(self) -> None:
        timestamp = _utc_now()
        for definition in list_observable_definitions():
            self.conn.execute(
                """
                INSERT INTO observables (observable_id, definition_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(observable_id) DO NOTHING
                """,
                (
                    definition.observable_id,
                    json.dumps(definition.to_document(), sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )

    def _seed_builtin_signal_specs(self) -> None:
        timestamp = _utc_now()
        specification_ids = (
            "momentum_1d",
            "momentum_3d",
            "momentum_5d",
            "reversal_1d",
            "reversal_3d",
            "reversal_5d",
            "average_gap_3d",
            "average_gap_5d",
            "range_position_5d",
        )
        for signal_id in specification_ids:
            definition = find_signal_spec(signal_id)
            if definition is None:
                continue
            self.conn.execute(
                """
                INSERT INTO signal_specs (
                    signal_id, target_id, definition_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(signal_id) DO NOTHING
                """,
                (
                    definition.signal_id,
                    definition.target_id,
                    json.dumps(definition.to_document(), sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )

    def get_target(self, target_id: str) -> TargetState | None:
        row = self.conn.execute(
            """
            SELECT target_id, definition_json
            FROM targets
            WHERE target_id = ?
            """,
            (target_id,),
        ).fetchone()
        return _row_to_target(row)

    def get_observable(self, observable_id: str) -> ObservableState | None:
        row = self.conn.execute(
            """
            SELECT observable_id, definition_json
            FROM observables
            WHERE observable_id = ?
            """,
            (observable_id,),
        ).fetchone()
        return _row_to_observable(row)

    def get_subject_set(self, subject_set_id: str) -> SubjectSetState | None:
        row = self.conn.execute(
            """
            SELECT subject_set_id, definition_json
            FROM subject_sets
            WHERE subject_set_id = ?
            """,
            (subject_set_id,),
        ).fetchone()
        return _row_to_subject_set(row)

    def get_signal_discovery_spec(
        self,
        signal_discovery_id: str,
    ) -> SignalDiscoverySpecState | None:
        row = self.conn.execute(
            """
            SELECT signal_discovery_id, definition_json
            FROM signal_discoveries
            WHERE signal_discovery_id = ?
            """,
            (signal_discovery_id,),
        ).fetchone()
        return _row_to_signal_discovery_spec(row)

    def get_evaluation_spec(
        self,
        evaluation_spec_id: str,
    ) -> EvaluationSpecState | None:
        row = self.conn.execute(
            """
            SELECT evaluation_spec_id, definition_json
            FROM evaluation_specs
            WHERE evaluation_spec_id = ?
            """,
            (evaluation_spec_id,),
        ).fetchone()
        return _row_to_evaluation_spec(row)

    def get_observation_frame_cache(self, cache_key: str) -> str | None:
        row = self.conn.execute(
            """
            SELECT frame_json
            FROM observation_frame_cache
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def list_observables(self, *, limit: int = 100) -> list[ObservableState]:
        rows = self.conn.execute(
            """
            SELECT observable_id, definition_json
            FROM observables
            ORDER BY observable_id ASC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [_row_to_observable(row) for row in rows if row is not None]

    def get_signal_spec(self, signal_id: str) -> SignalSpecState | None:
        row = self.conn.execute(
            """
            SELECT signal_id, target_id, definition_json
            FROM signal_specs
            WHERE signal_id = ?
            """,
            (signal_id,),
        ).fetchone()
        return _row_to_signal_spec(row)

    def list_signal_specs(self, *, limit: int = 100) -> list[SignalSpecState]:
        rows = self.conn.execute(
            """
            SELECT signal_id, target_id, definition_json
            FROM signal_specs
            ORDER BY signal_id ASC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [_row_to_signal_spec(row) for row in rows if row is not None]

    def upsert_subject_set(
        self,
        subject_set_id: str,
        *,
        definition: SubjectSet,
        recorded_at: str | None = None,
    ) -> SubjectSetState:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        definition_json = json.dumps(
            {
                "instruments": [
                    {
                        "instrument_id": item.instrument_id,
                        "instrument_type": item.instrument_type,
                        "asset": item.asset,
                        "venue": item.venue,
                        "quote_ccy": item.quote_ccy,
                        "contract_family": item.contract_family,
                        "asset_class": item.asset_class,
                        "region": item.region,
                        "liquidity_tier": item.liquidity_tier,
                        "cluster": item.cluster,
                        "expiry": item.expiry,
                        "roll_rule": item.roll_rule,
                        "multiplier": item.multiplier,
                        "margin_model": item.margin_model,
                    }
                    for item in definition.instruments
                ],
                "observation_specs": [
                    {
                        "observation_spec_id": item.observation_spec_id,
                        "observable_id": item.observable_id,
                        "adapter_kind": item.adapter_kind,
                        "source_id": item.source_id,
                        "resolution": item.resolution,
                        "provided_observable_ids": list(item.provided_observable_ids),
                    }
                    for item in definition.observation_specs
                ],
                "bindings": [
                    {
                        "subject_id": item.subject_id,
                        "subject_kind": item.subject_kind,
                        "asset": item.asset,
                        "observation_spec_id": item.observation_spec_id,
                        "instrument_id": item.instrument_id,
                    }
                    for item in definition.bindings
                ],
                "universe_policy": {
                    "base_currency": definition.universe_policy.base_currency,
                    "trading_calendar": definition.universe_policy.trading_calendar,
                    "benchmark_id": definition.universe_policy.benchmark_id,
                },
            },
            sort_keys=True,
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO subject_sets (
                    subject_set_id, definition_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(subject_set_id) DO UPDATE SET
                    definition_json = excluded.definition_json,
                    updated_at = excluded.updated_at
                """,
                (
                    subject_set_id,
                    definition_json,
                    timestamp,
                    timestamp,
                ),
            )
        state = self.get_subject_set(subject_set_id)
        assert state is not None
        return state

    def upsert_signal_discovery_spec(
        self,
        signal_discovery_id: str,
        *,
        definition: SignalDiscoverySpec,
        recorded_at: str | None = None,
    ) -> SignalDiscoverySpecState:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        definition_json = json.dumps(definition.to_document(), sort_keys=True)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO signal_discoveries (
                    signal_discovery_id, definition_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(signal_discovery_id) DO UPDATE SET
                    definition_json = excluded.definition_json,
                    updated_at = excluded.updated_at
                """,
                (
                    signal_discovery_id,
                    definition_json,
                    timestamp,
                    timestamp,
                ),
            )
        state = self.get_signal_discovery_spec(signal_discovery_id)
        assert state is not None
        return state

    def upsert_evaluation_spec(
        self,
        evaluation_spec_id: str,
        *,
        definition: EvaluationSpec,
        recorded_at: str | None = None,
    ) -> EvaluationSpecState:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        definition_json = json.dumps(definition.to_document(), sort_keys=True)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO evaluation_specs (
                    evaluation_spec_id, definition_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(evaluation_spec_id) DO UPDATE SET
                    definition_json = excluded.definition_json,
                    updated_at = excluded.updated_at
                """,
                (
                    evaluation_spec_id,
                    definition_json,
                    timestamp,
                    timestamp,
                ),
            )
        state = self.get_evaluation_spec(evaluation_spec_id)
        assert state is not None
        return state

    def upsert_observation_frame_cache(
        self,
        cache_key: str,
        *,
        frame_json: str,
        recorded_at: str | None = None,
    ) -> None:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO observation_frame_cache (
                    cache_key, frame_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    frame_json = excluded.frame_json,
                    updated_at = excluded.updated_at
                """,
                (
                    cache_key,
                    frame_json,
                    timestamp,
                    timestamp,
                ),
            )

    def list_subject_sets(self, *, limit: int = 20) -> list[SubjectSetState]:
        rows = self.conn.execute(
            """
            SELECT subject_set_id, definition_json
            FROM subject_sets
            ORDER BY subject_set_id ASC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [_row_to_subject_set(row) for row in rows if row is not None]

    def list_signal_discovery_specs(
        self,
        *,
        limit: int = 20,
    ) -> list[SignalDiscoverySpecState]:
        rows = self.conn.execute(
            """
            SELECT signal_discovery_id, definition_json
            FROM signal_discoveries
            ORDER BY signal_discovery_id ASC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [
            _row_to_signal_discovery_spec(row)
            for row in rows
            if row is not None
        ]

    def list_evaluation_specs(
        self,
        *,
        limit: int = 20,
    ) -> list[EvaluationSpecState]:
        rows = self.conn.execute(
            """
            SELECT evaluation_spec_id, definition_json
            FROM evaluation_specs
            ORDER BY evaluation_spec_id ASC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [_row_to_evaluation_spec(row) for row in rows if row is not None]

    def register_target(
        self,
        target_id: str,
        *,
        definition: TargetDefinition | None = None,
        recorded_at: str | None = None,
    ) -> TargetState:
        self.ensure_schema()
        existing = self.get_target(target_id)
        if existing is not None:
            return existing

        resolved_definition = definition or find_target_definition(target_id)
        if resolved_definition is None:
            raise ValueError(f"target definition must exist before use: {target_id}")
        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO targets (target_id, definition_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    resolved_definition.target_id,
                    json.dumps(resolved_definition.to_document(), sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )
        target_state = self.get_target(target_id)
        assert target_state is not None
        return target_state

    def register_observable(
        self,
        observable_id: str,
        *,
        definition: ObservableDefinition | None = None,
        recorded_at: str | None = None,
    ) -> tuple[ObservableState, bool]:
        self.ensure_schema()
        existing = self.get_observable(observable_id)
        if existing is not None:
            return existing, False

        resolved_definition = definition or find_observable_definition(observable_id)
        if resolved_definition is None:
            raise ValueError(
                f"observable definition must exist before use: {observable_id}"
            )
        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO observables (observable_id, definition_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    resolved_definition.observable_id,
                    json.dumps(resolved_definition.to_document(), sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )
        state = self.get_observable(observable_id)
        assert state is not None
        return state, True

    def register_signal_spec(
        self,
        signal_id: str,
        *,
        definition: SignalSpec | None = None,
        recorded_at: str | None = None,
    ) -> tuple[SignalSpecState, bool]:
        self.ensure_schema()
        existing = self.get_signal_spec(signal_id)
        if existing is not None:
            return existing, False

        resolved_definition = definition or find_signal_spec(signal_id)
        if resolved_definition is None:
            raise ValueError(
                f"signal spec definition must exist before use: {signal_id}"
            )
        timestamp = recorded_at or _utc_now()
        self.register_target(
            resolved_definition.target_id,
            definition=resolved_definition.target,
            recorded_at=timestamp,
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO signal_specs (
                    signal_id, target_id, definition_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    resolved_definition.signal_id,
                    resolved_definition.target_id,
                    json.dumps(resolved_definition.to_document(), sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )
        state = self.get_signal_spec(signal_id)
        assert state is not None
        return state, True

    def get_signal(
        self,
        signal_id: str,
    ) -> SignalState | None:
        row = self.conn.execute(
            """
            SELECT signal_id, specification_signal_id, subject_id, asset, target_id,
                   definition_json, status,
                   prediction_count, observation_count
            FROM signals
            WHERE signal_id = ?
            """,
            (signal_id,),
        ).fetchone()
        return _row_to_signal(row)

    def list_signals(
        self,
        *,
        subject_id: str | None = None,
        asset: str | None = None,
        target_id: str | None = DEFAULT_TARGET,
    ) -> list[SignalState]:
        resolved_asset = (
            default_runtime_asset(subject_id)
            if asset is None
            else asset
        )
        if subject_id is not None and target_id is None:
            rows = self.conn.execute(
                """
                SELECT signal_id, specification_signal_id, subject_id, asset, target_id,
                       definition_json, status,
                       prediction_count, observation_count
                FROM signals
                WHERE subject_id = ?
                ORDER BY target_id ASC, observation_count DESC, prediction_count DESC, signal_id ASC
                """,
                (subject_id,),
            ).fetchall()
        elif subject_id is not None:
            rows = self.conn.execute(
                """
                SELECT signal_id, specification_signal_id, subject_id, asset, target_id,
                       definition_json, status,
                       prediction_count, observation_count
                FROM signals
                WHERE subject_id = ? AND target_id = ?
                ORDER BY observation_count DESC, prediction_count DESC, signal_id ASC
                """,
                (subject_id, target_id),
            ).fetchall()
        elif target_id is None:
            rows = self.conn.execute(
                """
                SELECT signal_id, specification_signal_id, subject_id, asset, target_id,
                       definition_json, status,
                       prediction_count, observation_count
                FROM signals
                WHERE asset = ?
                ORDER BY target_id ASC, observation_count DESC, prediction_count DESC, signal_id ASC
                """,
                (resolved_asset,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT signal_id, specification_signal_id, subject_id, asset, target_id,
                       definition_json, status,
                       prediction_count, observation_count
                FROM signals
                WHERE asset = ? AND target_id = ?
                ORDER BY observation_count DESC, prediction_count DESC, signal_id ASC
                """,
                (resolved_asset, target_id),
            ).fetchall()
        return [_row_to_signal(row) for row in rows if row is not None]

    def get_signal_metric(
        self,
        signal_id: str,
    ) -> SignalMetricState | None:
        row = self.conn.execute(
            """
            SELECT signal_id, corr, mmc, sample_count, window_size,
                   mmc_baseline_type, mmc_peer_count, mmc_sample_count,
                   start_evaluation_id, end_evaluation_id, updated_at
            FROM signal_metrics
            WHERE signal_id = ?
            """,
            (signal_id,),
        ).fetchone()
        return _row_to_signal_metric(row)

    def list_signal_metrics(
        self,
        *,
        signal_ids: list[str] | None = None,
    ) -> list[SignalMetricState]:
        if not signal_ids:
            rows = self.conn.execute(
                """
                SELECT signal_id, corr, mmc, mmc_baseline_type, mmc_peer_count,
                       sample_count, mmc_sample_count, window_size,
                       start_evaluation_id, end_evaluation_id, updated_at
                FROM signal_metrics
                ORDER BY corr DESC, mmc DESC, signal_id ASC
                """
            ).fetchall()
            return [
                _row_to_signal_metric(row) for row in rows if row is not None
            ]

        placeholders = ", ".join("?" for _ in signal_ids)
        rows = self.conn.execute(
            f"""
            SELECT signal_id, corr, mmc, mmc_baseline_type, mmc_peer_count,
                   sample_count, mmc_sample_count, window_size,
                   start_evaluation_id, end_evaluation_id, updated_at
            FROM signal_metrics
            WHERE signal_id IN ({placeholders})
            ORDER BY corr DESC, mmc DESC, signal_id ASC
            """,
            tuple(signal_ids),
        ).fetchall()
        return [_row_to_signal_metric(row) for row in rows if row is not None]

    def upsert_meta_prediction(
        self,
        *,
        evaluation_id: str,
        subject_id: str | None = None,
        asset: str,
        target_id: str,
        aggregation_kind: str,
        value: float,
        contributor_count: int,
        details_json: str | None,
        recorded_at: str | None = None,
    ) -> None:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        resolved_subject_id = subject_id or asset
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO meta_predictions (
                    evaluation_id, subject_id, asset, target_id, aggregation_kind, value,
                    contributor_count, details_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_id, aggregation_kind) DO UPDATE SET
                    subject_id = excluded.subject_id,
                    asset = excluded.asset,
                    target_id = excluded.target_id,
                    value = excluded.value,
                    contributor_count = excluded.contributor_count,
                    details_json = excluded.details_json,
                    updated_at = excluded.updated_at
                """,
                (
                    evaluation_id,
                    resolved_subject_id,
                    asset,
                    target_id,
                    aggregation_kind,
                    float(value),
                    int(contributor_count),
                    details_json,
                    timestamp,
                    timestamp,
                ),
            )

    def list_meta_predictions(
        self,
        *,
        subject_id: str | None = None,
        asset: str | None = None,
        target_id: str | None = None,
        aggregation_kind: str | None = None,
        limit: int = 20,
    ) -> list[MetaPredictionState]:
        filters: list[str] = []
        params: list[Any] = []
        if subject_id is not None:
            filters.append("subject_id = ?")
            params.append(subject_id)
        else:
            resolved_asset = default_runtime_asset()
            filters.append("asset = ?")
            params.append(resolved_asset if asset is None else asset)
        if target_id is not None:
            filters.append("target_id = ?")
            params.append(target_id)
        if aggregation_kind is not None:
            filters.append("aggregation_kind = ?")
            params.append(aggregation_kind)
        params.append(max(int(limit), 1))
        rows = self.conn.execute(
            f"""
            SELECT evaluation_id, subject_id, asset, target_id, aggregation_kind, value,
                   contributor_count, details_json, created_at, updated_at
            FROM meta_predictions
            WHERE {' AND '.join(filters)}
            ORDER BY updated_at DESC, evaluation_id DESC, aggregation_kind ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_meta_prediction(row) for row in rows if row is not None]

    def upsert_meta_prediction_metric(
        self,
        *,
        aggregation_kind: str,
        subject_id: str | None = None,
        asset: str,
        target_id: str,
        corr: float,
        sample_count: int,
        window_size: int,
        start_evaluation_id: str | None,
        end_evaluation_id: str | None,
        recorded_at: str | None = None,
    ) -> None:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        resolved_subject_id = subject_id or asset
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO meta_prediction_metrics (
                    aggregation_kind, subject_id, asset, target_id, corr, sample_count,
                    window_size, start_evaluation_id, end_evaluation_id, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(aggregation_kind, subject_id, target_id) DO UPDATE SET
                    asset = excluded.asset,
                    corr = excluded.corr,
                    sample_count = excluded.sample_count,
                    window_size = excluded.window_size,
                    start_evaluation_id = excluded.start_evaluation_id,
                    end_evaluation_id = excluded.end_evaluation_id,
                    updated_at = excluded.updated_at
                """,
                (
                    aggregation_kind,
                    resolved_subject_id,
                    asset,
                    target_id,
                    float(corr),
                    int(sample_count),
                    int(window_size),
                    start_evaluation_id,
                    end_evaluation_id,
                    timestamp,
                ),
            )

    def list_meta_prediction_metrics(
        self,
        *,
        subject_id: str | None = None,
        asset: str | None = None,
        target_id: str | None = None,
    ) -> list[MetaPredictionMetricState]:
        filters: list[str] = []
        params: list[Any] = []
        if subject_id is not None:
            filters.append("subject_id = ?")
            params.append(subject_id)
        else:
            resolved_asset = default_runtime_asset()
            filters.append("asset = ?")
            params.append(resolved_asset if asset is None else asset)
        if target_id is not None:
            filters.append("target_id = ?")
            params.append(target_id)
        rows = self.conn.execute(
            f"""
            SELECT aggregation_kind, subject_id, asset, target_id, corr, sample_count,
                   window_size, start_evaluation_id, end_evaluation_id, updated_at
            FROM meta_prediction_metrics
            WHERE {' AND '.join(filters)}
            ORDER BY target_id ASC, corr DESC, aggregation_kind ASC
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_meta_prediction_metric(row) for row in rows if row is not None]

    def upsert_portfolio_decision(
        self,
        *,
        portfolio_id: str,
        subject_id: str,
        target_id: str,
        aggregation_kind: str,
        as_of: str,
        target_weight: float,
        position_delta: float,
        target_notional: float | None,
        target_quantity: float | None,
        entry_allowed: bool,
        risk_scale: float,
        details_json: str | None,
        recorded_at: str | None = None,
    ) -> None:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO portfolio_decisions (
                    portfolio_id, subject_id, target_id, aggregation_kind,
                    as_of, target_weight, position_delta, target_notional,
                    target_quantity, entry_allowed, risk_scale, details_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(portfolio_id, subject_id, target_id, aggregation_kind, as_of)
                DO UPDATE SET
                    target_weight = excluded.target_weight,
                    position_delta = excluded.position_delta,
                    target_notional = excluded.target_notional,
                    target_quantity = excluded.target_quantity,
                    entry_allowed = excluded.entry_allowed,
                    risk_scale = excluded.risk_scale,
                    details_json = excluded.details_json,
                    updated_at = excluded.updated_at
                """,
                (
                    portfolio_id,
                    subject_id,
                    target_id,
                    aggregation_kind,
                    as_of,
                    float(target_weight),
                    float(position_delta),
                    None if target_notional is None else float(target_notional),
                    None if target_quantity is None else float(target_quantity),
                    int(bool(entry_allowed)),
                    float(risk_scale),
                    details_json,
                    timestamp,
                    timestamp,
                ),
            )

    def list_portfolio_decisions(
        self,
        *,
        portfolio_id: str | None = None,
        target_id: str | None = None,
        aggregation_kind: str | None = None,
        limit: int = 20,
    ) -> list[PortfolioDecisionState]:
        filters: list[str] = []
        params: list[Any] = []
        if portfolio_id is not None:
            filters.append("portfolio_id = ?")
            params.append(portfolio_id)
        if target_id is not None:
            filters.append("target_id = ?")
            params.append(target_id)
        if aggregation_kind is not None:
            filters.append("aggregation_kind = ?")
            params.append(aggregation_kind)
        where_clause = ""
        if filters:
            where_clause = f"WHERE {' AND '.join(filters)}"
        params.append(max(int(limit), 1))
        rows = self.conn.execute(
            f"""
            SELECT portfolio_id, subject_id, target_id, aggregation_kind,
                   as_of, target_weight, position_delta, target_notional,
                   target_quantity, entry_allowed, risk_scale, details_json,
                   created_at, updated_at
            FROM portfolio_decisions
            {where_clause}
            ORDER BY updated_at DESC, as_of DESC, subject_id ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_portfolio_decision(row) for row in rows if row is not None]

    def get_latest_portfolio_decisions(
        self,
        *,
        portfolio_id: str,
        aggregation_kind: str,
    ) -> list[PortfolioDecisionState]:
        row = self.conn.execute(
            """
            SELECT MAX(as_of) AS latest_as_of
            FROM portfolio_decisions
            WHERE portfolio_id = ? AND aggregation_kind = ?
            """,
            (portfolio_id, aggregation_kind),
        ).fetchone()
        if row is None or row["latest_as_of"] is None:
            return []
        latest_as_of = str(row["latest_as_of"])
        rows = self.conn.execute(
            """
            SELECT portfolio_id, subject_id, target_id, aggregation_kind,
                   as_of, target_weight, position_delta, target_notional,
                   target_quantity, entry_allowed, risk_scale, details_json,
                   created_at, updated_at
            FROM portfolio_decisions
            WHERE portfolio_id = ? AND aggregation_kind = ? AND as_of = ?
            ORDER BY subject_id ASC
            """,
            (portfolio_id, aggregation_kind, latest_as_of),
        ).fetchall()
        return [_row_to_portfolio_decision(row) for row in rows if row is not None]

    def create_validation_run(
        self,
        *,
        run_id: str,
        spec_json: str,
        cross_instrument_contract: CrossInstrumentReportContract | None = None,
        validation_result_set: ValidationResultSet | None = None,
        recorded_at: str | None = None,
    ) -> None:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        criteria = (
            default_validation_result_set_cross_instrument_contract()
            if cross_instrument_contract is None
            else cross_instrument_contract
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO validation_runs (
                    run_id, spec_json, criteria_json, summary_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    spec_json,
                    json.dumps(criteria.to_document(), sort_keys=True),
                    (
                        None
                        if validation_result_set is None
                        else json.dumps(validation_result_set.to_document(), sort_keys=True)
                    ),
                    timestamp,
                ),
            )

    def get_validation_run(self, run_id: str) -> ValidationRunState | None:
        row = self.conn.execute(
            """
            SELECT run_id, spec_json, criteria_json, summary_json, created_at
            FROM validation_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        return _row_to_validation_run(row)

    def get_latest_validation_run(self) -> ValidationRunState | None:
        row = self.conn.execute(
            """
            SELECT run_id, spec_json, criteria_json, summary_json, created_at
            FROM validation_runs
            ORDER BY created_at DESC, run_id DESC
            LIMIT 1
            """
        ).fetchone()
        return _row_to_validation_run(row)

    def upsert_screening_result(
        self,
        *,
        result: ScreeningResult,
    ) -> ScreeningResultState:
        self.ensure_schema()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO screening_results (
                    screening_result_id, signal_discovery_id, result_json, created_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(screening_result_id) DO UPDATE SET
                    signal_discovery_id = excluded.signal_discovery_id,
                    result_json = excluded.result_json,
                    created_at = excluded.created_at
                """,
                (
                    result.screening_result_id,
                    result.signal_discovery_id,
                    json.dumps(result.to_document(), sort_keys=True),
                    result.created_at,
                ),
            )
        state = self.get_screening_result(result.screening_result_id)
        assert state is not None
        return state

    def get_screening_result(
        self,
        screening_result_id: str,
    ) -> ScreeningResultState | None:
        row = self.conn.execute(
            """
            SELECT screening_result_id, signal_discovery_id, result_json, created_at
            FROM screening_results
            WHERE screening_result_id = ?
            """,
            (screening_result_id,),
        ).fetchone()
        return _row_to_screening_result(row)

    def list_screening_results(
        self,
        *,
        signal_discovery_id: str | None = None,
        limit: int = 20,
    ) -> list[ScreeningResultState]:
        if signal_discovery_id is None:
            rows = self.conn.execute(
                """
                SELECT screening_result_id, signal_discovery_id, result_json, created_at
                FROM screening_results
                ORDER BY created_at DESC, screening_result_id DESC
                LIMIT ?
                """,
                (max(int(limit), 1),),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT screening_result_id, signal_discovery_id, result_json, created_at
                FROM screening_results
                WHERE signal_discovery_id = ?
                ORDER BY created_at DESC, screening_result_id DESC
                LIMIT ?
                """,
                (
                    signal_discovery_id,
                    max(int(limit), 1),
                ),
            ).fetchall()
        return [
            _row_to_screening_result(row)
            for row in rows
            if row is not None
        ]

    def upsert_compressed_belief(
        self,
        *,
        belief: CompressedBelief,
    ) -> CompressedBeliefState:
        self.ensure_schema()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO compressed_beliefs (
                    compressed_belief_id, signal_discovery_id, belief_json, created_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(compressed_belief_id) DO UPDATE SET
                    signal_discovery_id = excluded.signal_discovery_id,
                    belief_json = excluded.belief_json,
                    created_at = excluded.created_at
                """,
                (
                    belief.compressed_belief_id,
                    belief.signal_discovery_id,
                    json.dumps(belief.to_document(), sort_keys=True),
                    belief.created_at,
                ),
            )
        state = self.get_compressed_belief(belief.compressed_belief_id)
        assert state is not None
        return state

    def get_compressed_belief(
        self,
        compressed_belief_id: str,
    ) -> CompressedBeliefState | None:
        row = self.conn.execute(
            """
            SELECT compressed_belief_id, signal_discovery_id, belief_json, created_at
            FROM compressed_beliefs
            WHERE compressed_belief_id = ?
            """,
            (compressed_belief_id,),
        ).fetchone()
        return _row_to_compressed_belief(row)

    def list_compressed_beliefs(
        self,
        *,
        signal_discovery_id: str | None = None,
        limit: int = 20,
    ) -> list[CompressedBeliefState]:
        if signal_discovery_id is None:
            rows = self.conn.execute(
                """
                SELECT compressed_belief_id, signal_discovery_id, belief_json, created_at
                FROM compressed_beliefs
                ORDER BY created_at DESC, compressed_belief_id DESC
                LIMIT ?
                """,
                (max(int(limit), 1),),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT compressed_belief_id, signal_discovery_id, belief_json, created_at
                FROM compressed_beliefs
                WHERE signal_discovery_id = ?
                ORDER BY created_at DESC, compressed_belief_id DESC
                LIMIT ?
                """,
                (
                    signal_discovery_id,
                    max(int(limit), 1),
                ),
            ).fetchall()
        return [_row_to_compressed_belief(row) for row in rows if row is not None]

    def upsert_signal_discovery_run(
        self,
        *,
        run: SignalDiscoveryRun,
    ) -> SignalDiscoveryRunState:
        self.ensure_schema()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO signal_discovery_runs (
                    signal_discovery_run_id, signal_discovery_id, run_json, created_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(signal_discovery_run_id) DO UPDATE SET
                    signal_discovery_id = excluded.signal_discovery_id,
                    run_json = excluded.run_json,
                    created_at = excluded.created_at
                """,
                (
                    run.signal_discovery_run_id,
                    run.signal_discovery_id,
                    json.dumps(run.to_document(), sort_keys=True),
                    run.created_at,
                ),
            )
        state = self.get_signal_discovery_run(run.signal_discovery_run_id)
        assert state is not None
        return state

    def upsert_initial_strategy_state(
        self,
        *,
        state: InitialStrategyState,
    ) -> InitialStrategyStateRecord:
        self.ensure_schema()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO initial_strategy_states (
                    initial_strategy_state_id, strategy_id, signal_train_id, signal_discovery_id, artifact_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(initial_strategy_state_id) DO UPDATE SET
                    strategy_id = excluded.strategy_id,
                    signal_train_id = excluded.signal_train_id,
                    signal_discovery_id = excluded.signal_discovery_id,
                    artifact_json = excluded.artifact_json,
                    created_at = excluded.created_at
                """,
                (
                    state.initial_strategy_state_id,
                    state.strategy_id,
                    state.signal_train_id,
                    state.signal_discovery_id,
                    json.dumps(state.to_document(), sort_keys=True),
                    state.created_at,
                ),
            )
        persisted = self.get_initial_strategy_state(state.initial_strategy_state_id)
        assert persisted is not None
        return persisted

    def upsert_trading_strategy(
        self,
        *,
        trading_strategy: TradingStrategySpec,
    ) -> TradingStrategyState:
        self.ensure_schema()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO strategy_specs (
                    strategy_id, spec_json, created_at
                )
                VALUES (?, ?, ?)
                ON CONFLICT(strategy_id) DO UPDATE SET
                    spec_json = excluded.spec_json,
                    created_at = excluded.created_at
                """,
                (
                    trading_strategy.strategy_id,
                    json.dumps(
                        {"trading_strategy": trading_strategy.to_document()},
                        sort_keys=True,
                    ),
                    trading_strategy.created_at,
                ),
            )
        state = self.get_trading_strategy(trading_strategy.strategy_id)
        assert state is not None
        return state

    def upsert_evaluation_task(
        self,
        *,
        task: EvaluationTask,
    ) -> EvaluationTaskState:
        self.ensure_schema()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO evaluation_tasks (
                    evaluation_task_id, task_json, created_at
                )
                VALUES (?, ?, ?)
                ON CONFLICT(evaluation_task_id) DO UPDATE SET
                    task_json = excluded.task_json,
                    created_at = excluded.created_at
                """,
                (
                    task.evaluation_task_id,
                    json.dumps(task.to_document(), sort_keys=True),
                    _utc_now(),
                ),
            )
        state = self.get_evaluation_task(task.evaluation_task_id)
        assert state is not None
        return state

    def upsert_evaluation_job_spec(
        self,
        *,
        job_spec: EvaluationJobSpec,
    ) -> EvaluationJobSpecState:
        self.ensure_schema()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO evaluation_job_specs (
                    evaluation_task_id, job_spec_json, created_at
                )
                VALUES (?, ?, ?)
                ON CONFLICT(evaluation_task_id) DO UPDATE SET
                    job_spec_json = excluded.job_spec_json,
                    created_at = excluded.created_at
                """,
                (
                    job_spec.evaluation_task_id,
                    json.dumps(job_spec.to_document(), sort_keys=True),
                    _utc_now(),
                ),
            )
        state = self.get_evaluation_job_spec(job_spec.evaluation_task_id)
        assert state is not None
        return state

    def get_signal_discovery_run(
        self,
        signal_discovery_run_id: str,
    ) -> SignalDiscoveryRunState | None:
        row = self.conn.execute(
            """
            SELECT signal_discovery_run_id, signal_discovery_id, run_json, created_at
            FROM signal_discovery_runs
            WHERE signal_discovery_run_id = ?
            """,
            (signal_discovery_run_id,),
        ).fetchone()
        return _row_to_signal_discovery_run(row)

    def get_initial_strategy_state(
        self,
        initial_strategy_state_id: str,
    ) -> InitialStrategyStateRecord | None:
        row = self.conn.execute(
            """
            SELECT initial_strategy_state_id, strategy_id, signal_train_id, signal_discovery_id, artifact_json, created_at
            FROM initial_strategy_states
            WHERE initial_strategy_state_id = ?
            """,
            (initial_strategy_state_id,),
        ).fetchone()
        return _row_to_initial_strategy_state(row)

    def get_trading_strategy(
        self,
        strategy_id: str,
    ) -> TradingStrategyState | None:
        row = self.conn.execute(
            """
            SELECT strategy_id, spec_json, created_at
            FROM strategy_specs
            WHERE strategy_id = ?
            """,
            (strategy_id,),
        ).fetchone()
        return _row_to_trading_strategy(row)

    def get_evaluation_task(
        self,
        evaluation_task_id: str,
    ) -> EvaluationTaskState | None:
        row = self.conn.execute(
            """
            SELECT evaluation_task_id, task_json, created_at
            FROM evaluation_tasks
            WHERE evaluation_task_id = ?
            """,
            (evaluation_task_id,),
        ).fetchone()
        return _row_to_evaluation_task(row)

    def get_evaluation_job_spec(
        self,
        evaluation_task_id: str,
    ) -> EvaluationJobSpecState | None:
        row = self.conn.execute(
            """
            SELECT evaluation_task_id, job_spec_json, created_at
            FROM evaluation_job_specs
            WHERE evaluation_task_id = ?
            """,
            (evaluation_task_id,),
        ).fetchone()
        return _row_to_evaluation_job_spec(row)

    def list_signal_discovery_runs(
        self,
        *,
        signal_discovery_id: str | None = None,
        execution_start_date: str | None = None,
        execution_end_date: str | None = None,
        limit: int = 20,
    ) -> list[SignalDiscoveryRunState]:
        filters = []
        params: list[object] = []
        if signal_discovery_id is not None:
            filters.append("signal_discovery_id = ?")
            params.append(signal_discovery_id)
        if execution_start_date is not None:
            filters.append("json_extract(run_json, '$.execution_start_date') = ?")
            params.append(execution_start_date)
        if execution_end_date is not None:
            filters.append("json_extract(run_json, '$.execution_end_date') = ?")
            params.append(execution_end_date)
        where_clause = ""
        if filters:
            where_clause = f"WHERE {' AND '.join(filters)}"
        params.append(max(int(limit), 1))
        rows = self.conn.execute(
            f"""
            SELECT signal_discovery_run_id, signal_discovery_id, run_json, created_at
            FROM signal_discovery_runs
            {where_clause}
            ORDER BY created_at DESC, signal_discovery_run_id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [
            _row_to_signal_discovery_run(row) for row in rows if row is not None
        ]

    def list_initial_strategy_states(
        self,
        *,
        strategy_id: str | None = None,
        signal_train_id: str | None = None,
        signal_discovery_id: str | None = None,
        fold_label: str | None = None,
        execution_start_date: str | None = None,
        execution_end_date: str | None = None,
        limit: int = 20,
    ) -> list[InitialStrategyStateRecord]:
        filters = []
        params: list[object] = []
        if strategy_id is not None:
            filters.append("strategy_id = ?")
            params.append(strategy_id)
        if signal_train_id is not None:
            filters.append("signal_train_id = ?")
            params.append(signal_train_id)
        if signal_discovery_id is not None:
            filters.append("signal_discovery_id = ?")
            params.append(signal_discovery_id)
        if fold_label is not None:
            filters.append("json_extract(artifact_json, '$.fold_label') = ?")
            params.append(fold_label)
        if execution_start_date is not None:
            filters.append("json_extract(artifact_json, '$.execution_start_date') = ?")
            params.append(execution_start_date)
        if execution_end_date is not None:
            filters.append("json_extract(artifact_json, '$.execution_end_date') = ?")
            params.append(execution_end_date)
        where_clause = ""
        if filters:
            where_clause = f"WHERE {' AND '.join(filters)}"
        params.append(max(int(limit), 1))
        rows = self.conn.execute(
            f"""
            SELECT initial_strategy_state_id, strategy_id, signal_train_id, signal_discovery_id, artifact_json, created_at
            FROM initial_strategy_states
            {where_clause}
            ORDER BY created_at DESC, initial_strategy_state_id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [
            _row_to_initial_strategy_state(row)
            for row in rows
            if row is not None
        ]

    def list_trading_strategies(
        self,
        *,
        limit: int = 20,
    ) -> list[TradingStrategyState]:
        rows = self.conn.execute(
            """
            SELECT strategy_id, spec_json, created_at
            FROM strategy_specs
            ORDER BY created_at DESC, strategy_id DESC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [_row_to_trading_strategy(row) for row in rows if row is not None]

    def list_evaluation_tasks(
        self,
        *,
        strategy_id: str | None = None,
        evaluation_spec_id: str | None = None,
        limit: int = 20,
    ) -> list[EvaluationTaskState]:
        filters = []
        params: list[object] = []
        if strategy_id is not None:
            filters.append("json_extract(task_json, '$.strategy_id') = ?")
            params.append(strategy_id)
        if evaluation_spec_id is not None:
            filters.append("json_extract(task_json, '$.evaluation_spec_id') = ?")
            params.append(evaluation_spec_id)
        where_clause = ""
        if filters:
            where_clause = f"WHERE {' AND '.join(filters)}"
        params.append(max(int(limit), 1))
        rows = self.conn.execute(
            f"""
            SELECT evaluation_task_id, task_json, created_at
            FROM evaluation_tasks
            {where_clause}
            ORDER BY created_at DESC, evaluation_task_id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_evaluation_task(row) for row in rows if row is not None]

    def upsert_evaluation_report(
        self,
        *,
        report: EvaluationReport,
    ) -> EvaluationReportState:
        self.ensure_schema()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO evaluation_reports (
                    evaluation_report_id, evaluation_spec_id, report_json, created_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(evaluation_report_id) DO UPDATE SET
                    evaluation_spec_id = excluded.evaluation_spec_id,
                    report_json = excluded.report_json,
                    created_at = excluded.created_at
                """,
                (
                    report.evaluation_report_id,
                    report.evaluation_spec_id,
                    json.dumps(report.to_document(), sort_keys=True),
                    report.created_at,
                ),
            )
        state = self.get_evaluation_report(report.evaluation_report_id)
        assert state is not None
        return state

    def upsert_evaluation_decision_trace(
        self,
        *,
        evaluation_report_id: str,
        evaluation_task_id: str | None = None,
        evaluation_fold_label: str,
        evaluation_range_label: str,
        result: Any,
        variant: str = "selected",
        step_granularity: str = "1d",
        subject_metadata_by_subject: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.ensure_schema()
        if evaluation_task_id is None:
            raise ValueError("evaluation decision trace requires evaluation_task_id")
        created_at = _utc_now()
        subject_metadata_by_subject = subject_metadata_by_subject or {}
        key = (
            evaluation_report_id,
                    evaluation_task_id,
            evaluation_fold_label,
            evaluation_range_label,
            variant,
        )
        with self.conn:
            self.conn.execute(
                """
                DELETE FROM evaluation_decision_trace_subject_steps
                WHERE evaluation_report_id = ?
                  AND evaluation_task_id = ?
                  AND evaluation_fold_label = ?
                  AND evaluation_range_label = ?
                  AND variant = ?
                """,
                key,
            )
            self.conn.execute(
                """
                DELETE FROM evaluation_decision_trace_steps
                WHERE evaluation_report_id = ?
                  AND evaluation_task_id = ?
                  AND evaluation_fold_label = ?
                  AND evaluation_range_label = ?
                  AND variant = ?
                """,
                key,
            )
            for step_index, step in enumerate(result.steps):
                self.conn.execute(
                    """
                    INSERT INTO evaluation_decision_trace_steps (
                        evaluation_report_id, evaluation_task_id, evaluation_fold_label,
                        evaluation_range_label, variant, step_index, step_as_of,
                        step_granularity, target_id, subject_set_id, gross_return,
                        net_return, gross_pnl_notional, net_pnl_notional, turnover,
                        traded_notional, cost_notional, funding_cost_notional,
                        borrow_cost_notional, roll_cost_notional,
                        gross_leverage_exposure, net_leverage_exposure,
                        long_leverage_exposure, short_leverage_exposure,
                        gross_notional_exposure, net_notional_exposure,
                        long_notional_exposure, short_notional_exposure,
                        gross_equity, net_equity, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        *key,
                        step_index,
                        step.date,
                        step_granularity,
                        result.target_id,
                        result.subject_set_id or "",
                        step.gross_return,
                        step.net_return,
                        step.gross_pnl_notional,
                        step.net_pnl_notional,
                        step.turnover,
                        step.traded_notional,
                        step.cost_notional,
                        step.funding_cost_notional,
                        step.borrow_cost_notional,
                        step.roll_cost_notional,
                        step.gross_leverage_exposure,
                        step.net_leverage_exposure,
                        step.long_leverage_exposure,
                        step.short_leverage_exposure,
                        step.gross_notional_exposure,
                        step.net_notional_exposure,
                        step.long_notional_exposure,
                        step.short_notional_exposure,
                        step.gross_equity,
                        step.net_equity,
                        created_at,
                    ),
                )
                for subject_step in step.subject_steps:
                    metadata = subject_metadata_by_subject.get(subject_step.subject_id, {})
                    self.conn.execute(
                        """
                        INSERT INTO evaluation_decision_trace_subject_steps (
                            evaluation_report_id, evaluation_task_id,
                            evaluation_fold_label, evaluation_range_label, variant,
                            step_index, subject_id, asset_class, cluster, signal_value,
                            realized_return, target_weight, position_delta,
                            target_notional, traded_notional, gross_pnl_notional,
                            execution_cost_notional, funding_cost_notional,
                            borrow_cost_notional, roll_cost_notional, cost_notional,
                            net_pnl_notional, net_return_contribution, risk_scale,
                            entry_allowed, funding_cost_bps, borrow_fee_bps,
                            roll_cost_bps, contract_multiplier, target_contracts,
                            traded_contracts, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            *key,
                            step_index,
                            subject_step.subject_id,
                            metadata.get("asset_class"),
                            metadata.get("cluster"),
                            subject_step.signal_value,
                            subject_step.realized_return,
                            subject_step.target_weight,
                            subject_step.position_delta,
                            subject_step.target_notional,
                            subject_step.traded_notional,
                            subject_step.gross_pnl_notional,
                            subject_step.execution_cost_notional,
                            subject_step.funding_cost_notional,
                            subject_step.borrow_cost_notional,
                            subject_step.roll_cost_notional,
                            subject_step.cost_notional,
                            subject_step.net_pnl_notional,
                            subject_step.net_return_contribution,
                            subject_step.risk_scale,
                            int(subject_step.entry_allowed),
                            subject_step.funding_cost_bps,
                            subject_step.borrow_fee_bps,
                            subject_step.roll_cost_bps,
                            subject_step.contract_multiplier,
                            subject_step.target_contracts,
                            subject_step.traded_contracts,
                            created_at,
                        ),
                    )

    def list_evaluation_decision_trace_steps(
        self,
        *,
        evaluation_report_id: str | None = None,
        evaluation_task_id: str | None = None,
        evaluation_fold_label: str | None = None,
        evaluation_range_label: str | None = None,
        variant: str | None = None,
        limit: int = 100,
    ) -> list[EvaluationDecisionTraceStepState]:
        self.ensure_schema()
        filters: list[str] = []
        params: list[object] = []
        for column, value in (
            ("evaluation_report_id", evaluation_report_id),
            ("evaluation_task_id", evaluation_task_id),
            ("evaluation_fold_label", evaluation_fold_label),
            ("evaluation_range_label", evaluation_range_label),
            ("variant", variant),
        ):
            if value is not None:
                filters.append(f"{column} = ?")
                params.append(value)
        where_clause = "" if not filters else f"WHERE {' AND '.join(filters)}"
        params.append(max(int(limit), 1))
        rows = self.conn.execute(
            f"""
            SELECT *
            FROM evaluation_decision_trace_steps
            {where_clause}
            ORDER BY evaluation_report_id, evaluation_task_id,
                     evaluation_fold_label, evaluation_range_label, variant, step_index
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [
            item
            for row in rows
            if (item := _row_to_evaluation_decision_trace_step(row)) is not None
        ]

    def list_evaluation_decision_trace_subject_steps(
        self,
        *,
        evaluation_report_id: str | None = None,
        evaluation_task_id: str | None = None,
        evaluation_fold_label: str | None = None,
        evaluation_range_label: str | None = None,
        variant: str | None = None,
        subject_id: str | None = None,
        limit: int = 100,
    ) -> list[EvaluationDecisionTraceSubjectStepState]:
        self.ensure_schema()
        filters: list[str] = []
        params: list[object] = []
        for column, value in (
            ("evaluation_report_id", evaluation_report_id),
            ("evaluation_task_id", evaluation_task_id),
            ("evaluation_fold_label", evaluation_fold_label),
            ("evaluation_range_label", evaluation_range_label),
            ("variant", variant),
            ("subject_id", subject_id),
        ):
            if value is not None:
                filters.append(f"{column} = ?")
                params.append(value)
        where_clause = "" if not filters else f"WHERE {' AND '.join(filters)}"
        params.append(max(int(limit), 1))
        rows = self.conn.execute(
            f"""
            SELECT *
            FROM evaluation_decision_trace_subject_steps
            {where_clause}
            ORDER BY evaluation_report_id, evaluation_task_id,
                     evaluation_fold_label, evaluation_range_label, variant,
                     step_index, subject_id
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [
            item
            for row in rows
            if (item := _row_to_evaluation_decision_trace_subject_step(row))
            is not None
        ]

    def upsert_strategy_adaptation_state(
        self,
        *,
        state: StrategyAdaptationState,
    ) -> StrategyAdaptationStateRecord:
        self.ensure_schema()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO strategy_adaptation_states (
                    strategy_id, signal_discovery_id, signal_train_id, state_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(strategy_id) DO UPDATE SET
                    signal_discovery_id = excluded.signal_discovery_id,
                    signal_train_id = excluded.signal_train_id,
                    state_json = excluded.state_json,
                    created_at = excluded.created_at
                """,
                (
                    state.strategy_id,
                    state.signal_discovery_id,
                    state.signal_train_id,
                    json.dumps(state.to_document(), sort_keys=True),
                    state.created_at,
                ),
            )
        persisted = self.get_strategy_adaptation_state(state.strategy_id)
        assert persisted is not None
        return persisted

    def get_evaluation_report(
        self,
        evaluation_report_id: str,
    ) -> EvaluationReportState | None:
        row = self.conn.execute(
            """
            SELECT evaluation_report_id, evaluation_spec_id, report_json, created_at
            FROM evaluation_reports
            WHERE evaluation_report_id = ?
            """,
            (evaluation_report_id,),
        ).fetchone()
        return _row_to_evaluation_report(row)

    def get_latest_evaluation_report(self) -> EvaluationReportState | None:
        row = self.conn.execute(
            """
            SELECT evaluation_report_id, evaluation_spec_id, report_json, created_at
            FROM evaluation_reports
            ORDER BY created_at DESC, evaluation_report_id DESC
            LIMIT 1
            """
        ).fetchone()
        return _row_to_evaluation_report(row)

    def get_strategy_adaptation_state(
        self,
        strategy_id: str,
    ) -> StrategyAdaptationStateRecord | None:
        row = self.conn.execute(
            """
            SELECT strategy_id, signal_discovery_id, state_json, created_at
            FROM strategy_adaptation_states
            WHERE strategy_id = ?
            """,
            (strategy_id,),
        ).fetchone()
        return _row_to_strategy_adaptation_state(row)

    def list_evaluation_reports(
        self,
        *,
        evaluation_spec_id: str | None = None,
        limit: int = 20,
    ) -> list[EvaluationReportState]:
        if evaluation_spec_id is None:
            rows = self.conn.execute(
                """
                SELECT evaluation_report_id, evaluation_spec_id, report_json, created_at
                FROM evaluation_reports
                ORDER BY created_at DESC, evaluation_report_id DESC
                LIMIT ?
                """,
                (max(int(limit), 1),),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT evaluation_report_id, evaluation_spec_id, report_json, created_at
                FROM evaluation_reports
                WHERE evaluation_spec_id = ?
                ORDER BY created_at DESC, evaluation_report_id DESC
                LIMIT ?
                """,
                (
                    evaluation_spec_id,
                    max(int(limit), 1),
                ),
            ).fetchall()
        return [_row_to_evaluation_report(row) for row in rows if row is not None]

    def list_strategy_adaptation_states(
        self,
        *,
        strategy_id: str | None = None,
        signal_discovery_id: str | None = None,
        limit: int = 20,
    ) -> list[StrategyAdaptationStateRecord]:
        filters = []
        params: list[object] = []
        if strategy_id is not None:
            filters.append("strategy_id = ?")
            params.append(strategy_id)
        if signal_discovery_id is not None:
            filters.append("signal_discovery_id = ?")
            params.append(signal_discovery_id)
        where_clause = ""
        if filters:
            where_clause = f"WHERE {' AND '.join(filters)}"
        params.append(max(int(limit), 1))
        rows = self.conn.execute(
            f"""
            SELECT strategy_id, signal_discovery_id, state_json, created_at
            FROM strategy_adaptation_states
            {where_clause}
            ORDER BY created_at DESC, strategy_id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_strategy_adaptation_state(row) for row in rows if row is not None]

    def upsert_validation_signal_result(
        self,
        *,
        run_id: str,
        date_range_label: str,
        start_date: str,
        end_date: str,
        target_id: str,
        signal_id: str,
        window_size: int,
        corr: float,
        mmc: float | None,
        sample_count: int,
        mmc_sample_count: int,
        mmc_peer_count: int,
        mmc_baseline_type: str | None,
        recorded_at: str | None = None,
    ) -> None:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO validation_signal_results (
                    run_id, date_range_label, start_date, end_date, target_id,
                    signal_id, window_size, corr, mmc, sample_count,
                    mmc_sample_count, mmc_peer_count, mmc_baseline_type, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, date_range_label, target_id, signal_id, window_size)
                DO UPDATE SET
                    corr = excluded.corr,
                    mmc = excluded.mmc,
                    sample_count = excluded.sample_count,
                    mmc_sample_count = excluded.mmc_sample_count,
                    mmc_peer_count = excluded.mmc_peer_count,
                    mmc_baseline_type = excluded.mmc_baseline_type,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    date_range_label,
                    start_date,
                    end_date,
                    target_id,
                    signal_id,
                    int(window_size),
                    float(corr),
                    None if mmc is None else float(mmc),
                    int(sample_count),
                    int(mmc_sample_count),
                    int(mmc_peer_count),
                    mmc_baseline_type,
                    timestamp,
                ),
            )

    def list_validation_signal_results(
        self,
        *,
        run_id: str,
    ) -> list[ValidationSignalResultState]:
        rows = self.conn.execute(
            """
            SELECT run_id, date_range_label, start_date, end_date, target_id,
                   signal_id, window_size, corr, mmc, sample_count,
                   mmc_sample_count, mmc_peer_count, mmc_baseline_type, updated_at
            FROM validation_signal_results
            WHERE run_id = ?
            ORDER BY date_range_label ASC, target_id ASC, window_size ASC, corr DESC, signal_id ASC
            """,
            (run_id,),
        ).fetchall()
        return [
            _row_to_validation_signal_result(row)
            for row in rows
            if row is not None
        ]

    def upsert_validation_meta_result(
        self,
        *,
        run_id: str,
        date_range_label: str,
        start_date: str,
        end_date: str,
        target_id: str,
        aggregation_kind: str,
        window_size: int,
        corr: float,
        sample_count: int,
        recorded_at: str | None = None,
    ) -> None:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO validation_meta_results (
                    run_id, date_range_label, start_date, end_date, target_id,
                    aggregation_kind, window_size, corr, sample_count, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, date_range_label, target_id, aggregation_kind, window_size)
                DO UPDATE SET
                    corr = excluded.corr,
                    sample_count = excluded.sample_count,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    date_range_label,
                    start_date,
                    end_date,
                    target_id,
                    aggregation_kind,
                    int(window_size),
                    float(corr),
                    int(sample_count),
                    timestamp,
                ),
            )

    def list_validation_meta_results(
        self,
        *,
        run_id: str,
    ) -> list[ValidationMetaResultState]:
        rows = self.conn.execute(
            """
            SELECT run_id, date_range_label, start_date, end_date, target_id,
                   aggregation_kind, window_size, corr, sample_count, updated_at
            FROM validation_meta_results
            WHERE run_id = ?
            ORDER BY date_range_label ASC, target_id ASC, window_size ASC, corr DESC, aggregation_kind ASC
            """,
            (run_id,),
        ).fetchall()
        return [
            _row_to_validation_meta_result(row)
            for row in rows
            if row is not None
        ]

    def upsert_validation_decision_result(
        self,
        *,
        run_id: str,
        date_range_label: str,
        start_date: str,
        end_date: str,
        target_id: str,
        subject_set_id: str,
        aggregation_kind: str,
        window_size: int,
        gross_return_total: float,
        net_return_total: float,
        max_drawdown: float,
        mean_turnover: float,
        mean_gross_notional_exposure: float,
        mean_net_notional_exposure: float,
        mean_long_notional_exposure: float,
        mean_short_notional_exposure: float,
        mean_traded_notional: float,
        cost_notional_total: float,
        funding_cost_notional_total: float,
        borrow_cost_notional_total: float,
        roll_cost_notional_total: float,
        step_count: int,
        recorded_at: str | None = None,
    ) -> None:
        self.ensure_schema()
        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO validation_decision_results (
                    run_id, date_range_label, start_date, end_date, target_id,
                    subject_set_id, aggregation_kind, window_size, gross_return_total,
                    net_return_total, max_drawdown, mean_turnover,
                    mean_gross_notional_exposure, mean_net_notional_exposure,
                    mean_long_notional_exposure, mean_short_notional_exposure,
                    mean_traded_notional, cost_notional_total,
                    funding_cost_notional_total, borrow_cost_notional_total,
                    roll_cost_notional_total, step_count, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    run_id, date_range_label, target_id, subject_set_id,
                    aggregation_kind, window_size
                )
                DO UPDATE SET
                    gross_return_total = excluded.gross_return_total,
                    net_return_total = excluded.net_return_total,
                    max_drawdown = excluded.max_drawdown,
                    mean_turnover = excluded.mean_turnover,
                    mean_gross_notional_exposure = excluded.mean_gross_notional_exposure,
                    mean_net_notional_exposure = excluded.mean_net_notional_exposure,
                    mean_long_notional_exposure = excluded.mean_long_notional_exposure,
                    mean_short_notional_exposure = excluded.mean_short_notional_exposure,
                    mean_traded_notional = excluded.mean_traded_notional,
                    cost_notional_total = excluded.cost_notional_total,
                    funding_cost_notional_total = excluded.funding_cost_notional_total,
                    borrow_cost_notional_total = excluded.borrow_cost_notional_total,
                    roll_cost_notional_total = excluded.roll_cost_notional_total,
                    step_count = excluded.step_count,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    date_range_label,
                    start_date,
                    end_date,
                    target_id,
                    subject_set_id,
                    aggregation_kind,
                    int(window_size),
                    float(gross_return_total),
                    float(net_return_total),
                    float(max_drawdown),
                    float(mean_turnover),
                    float(mean_gross_notional_exposure),
                    float(mean_net_notional_exposure),
                    float(mean_long_notional_exposure),
                    float(mean_short_notional_exposure),
                    float(mean_traded_notional),
                    float(cost_notional_total),
                    float(funding_cost_notional_total),
                    float(borrow_cost_notional_total),
                    float(roll_cost_notional_total),
                    int(step_count),
                    timestamp,
                ),
            )

    def list_validation_decision_results(
        self,
        *,
        run_id: str,
    ) -> list[ValidationDecisionResultState]:
        rows = self.conn.execute(
            """
            SELECT run_id, date_range_label, start_date, end_date, target_id,
                   subject_set_id, aggregation_kind, window_size, gross_return_total,
                   net_return_total, max_drawdown, mean_turnover,
                   mean_gross_notional_exposure, mean_net_notional_exposure,
                   mean_long_notional_exposure, mean_short_notional_exposure,
                   mean_traded_notional, cost_notional_total,
                   funding_cost_notional_total, borrow_cost_notional_total,
                   roll_cost_notional_total, step_count, updated_at
            FROM validation_decision_results
            WHERE run_id = ?
            ORDER BY date_range_label ASC, target_id ASC, subject_set_id ASC,
                     window_size ASC, net_return_total DESC, aggregation_kind ASC
            """,
            (run_id,),
        ).fetchall()
        return [
            _row_to_validation_decision_result(row)
            for row in rows
            if row is not None
        ]

    def set_signal_status(
        self,
        signal_id: str,
        *,
        action: str,
        recorded_at: str | None = None,
    ) -> SignalState:
        self.ensure_schema()
        signal = self.get_signal(signal_id)
        if signal is None:
            raise ValueError(f"signal does not exist: {signal_id}")
        decision = decide_operator_transition(
            current_status=signal.status,
            action=action,
        )

        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                UPDATE signals
                SET status = ?, updated_at = ?
                WHERE signal_id = ?
                """,
                (decision.next_status, timestamp, signal_id),
            )

        updated = self.get_signal(signal_id)
        assert updated is not None
        return updated

    def register_signal(
        self,
        signal_id: str,
        *,
        recorded_at: str | None = None,
        asset: str | None = None,
        target_id: str = DEFAULT_TARGET,
        definition: SignalDefinition | None = None,
        specification_signal_id: str | None = None,
    ) -> tuple[SignalState, bool]:
        self.ensure_schema()
        existing = self.get_signal(signal_id)
        if existing is not None:
            return existing, False

        timestamp = recorded_at or _utc_now()
        resolved_definition = definition or find_signal_definition(signal_id)
        resolved_specification_signal_id = specification_signal_id
        if resolved_specification_signal_id is None:
            specification_definition = find_signal_spec(
                signal_id
            )
            if specification_definition is not None:
                resolved_specification_signal_id = (
                    specification_definition.signal_id
                )
        if (
            resolved_definition is not None
            and target_id != DEFAULT_TARGET
            and target_id != resolved_definition.target_id
        ):
            raise ValueError(
                "built-in signal target does not match provided target: "
                f"{target_id} != {resolved_definition.target_id}"
            )
        definition_json = (
            None
            if resolved_definition is None
            else json.dumps(resolved_definition.to_document(), sort_keys=True)
        )
        resolved_asset = (
            default_runtime_asset()
            if asset is None and resolved_definition is None
            else asset
        )
        resolved_asset = (
            resolved_definition.asset
            if resolved_definition is not None
            else str(resolved_asset)
        )
        resolved_subject_id = subject_id_for_signal(
            signal_id=signal_id,
            asset=resolved_asset,
        )
        resolved_target_id = (
            target_id if resolved_definition is None else resolved_definition.target_id
        )
        self.register_target(
            resolved_target_id,
            definition=None if resolved_definition is None else resolved_definition.target,
            recorded_at=timestamp,
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO signals (
                    signal_id, specification_signal_id, subject_id, asset, target_id,
                    definition_json, status,
                    prediction_count, observation_count, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'active', 0, 0, ?, ?)
                """,
                (
                    signal_id,
                    resolved_specification_signal_id,
                    resolved_subject_id,
                    resolved_asset,
                    resolved_target_id,
                    definition_json,
                    timestamp,
                    timestamp,
                ),
            )

        signal = self.get_signal(signal_id)
        assert signal is not None
        return signal, True

    def get_evaluation_snapshot(
        self,
        evaluation_id: str,
        signal_id: str,
    ) -> EvaluationSnapshot | None:
        row = self.conn.execute(
            """
            SELECT evaluation_id, subject_id, asset, target_id, signal_id, prediction_value,
                   observation_value, signed_edge, absolute_error, input_source,
                   input_range_start, input_range_end,
                   funding_cost_bps, borrow_fee_bps, roll_cost_bps, financing_cost_bps,
                   contract_multiplier, contract_id, contract_family, quote_ccy,
                   collateral_ccy, roll_event_json,
                   observation_spec_id, observable_id, adapter_kind,
                   signal_name, created_at
            FROM evaluation_snapshots
            WHERE evaluation_id = ? AND signal_id = ?
            """,
            (evaluation_id, signal_id),
        ).fetchone()
        return _row_to_snapshot(row)

    def get_prediction(
        self,
        evaluation_id: str,
        signal_id: str,
    ) -> PredictionRecord | None:
        row = self.conn.execute(
            """
            SELECT evaluation_id, signal_id, subject_id, asset, target_id, value, recorded_at
            FROM predictions
            WHERE evaluation_id = ? AND signal_id = ?
            """,
            (evaluation_id, signal_id),
        ).fetchone()
        return _row_to_prediction(row)

    def get_observation(self, evaluation_id: str) -> ObservationRecord | None:
        row = self.conn.execute(
            """
            SELECT evaluation_id, subject_id, asset, target_id, value, recorded_at
            FROM observations
            WHERE evaluation_id = ?
            """,
            (evaluation_id,),
        ).fetchone()
        return _row_to_observation(row)

    def list_observations_for_subject_or_asset(
        self,
        *,
        subject_id: str,
        asset: str,
        target_id: str,
        limit: int,
    ) -> list[ObservationRecord]:
        """List recent observations, preferring subject rows before asset fallback."""
        rows = self.conn.execute(
            """
            SELECT evaluation_id, subject_id, asset, target_id, value, recorded_at
            FROM observations
            WHERE subject_id = ? AND target_id = ?
            ORDER BY evaluation_id DESC
            LIMIT ?
            """,
            (subject_id, target_id, int(limit)),
        ).fetchall()
        if not rows:
            rows = self.conn.execute(
                """
                SELECT evaluation_id, subject_id, asset, target_id, value, recorded_at
                FROM observations
                WHERE asset = ? AND target_id = ?
                ORDER BY evaluation_id DESC
                LIMIT ?
                """,
                (asset, target_id, int(limit)),
            ).fetchall()
        return [
            observation
            for observation in (_row_to_observation(row) for row in rows)
            if observation is not None
        ]

    def list_evaluation_snapshots(self, *, limit: int = 20) -> list[EvaluationSnapshot]:
        rows = self.conn.execute(
            """
            SELECT evaluation_id, subject_id, asset, target_id, signal_id, prediction_value,
                   observation_value, signed_edge, absolute_error, input_source,
                   input_range_start, input_range_end,
                   funding_cost_bps, borrow_fee_bps, roll_cost_bps, financing_cost_bps,
                   contract_multiplier, contract_id, contract_family, quote_ccy,
                   collateral_ccy, roll_event_json,
                   observation_spec_id, observable_id, adapter_kind,
                   signal_name, created_at
            FROM evaluation_snapshots
            ORDER BY created_at DESC, evaluation_id DESC, signal_id DESC
            LIMIT ?
            """,
            (max(int(limit), 1),),
        ).fetchall()
        return [_row_to_snapshot(row) for row in rows]

    def list_latest_evaluation_snapshots(
        self,
        *,
        signal_ids: list[str],
    ) -> list[EvaluationSnapshot]:
        if not signal_ids:
            return []
        placeholders = ", ".join("?" for _ in signal_ids)
        rows = self.conn.execute(
            f"""
            SELECT evaluation_id, subject_id, asset, target_id, signal_id, prediction_value,
                   observation_value, signed_edge, absolute_error, input_source,
                   input_range_start, input_range_end,
                   funding_cost_bps, borrow_fee_bps, roll_cost_bps, financing_cost_bps,
                   contract_multiplier, contract_id, contract_family, quote_ccy,
                   collateral_ccy, roll_event_json,
                   observation_spec_id, observable_id, adapter_kind,
                   signal_name, created_at
            FROM evaluation_snapshots
            WHERE signal_id IN ({placeholders})
            ORDER BY created_at DESC, evaluation_id DESC, signal_id ASC
            """,
            tuple(signal_ids),
        ).fetchall()
        latest_by_signal_id: dict[str, EvaluationSnapshot] = {}
        for row in rows:
            snapshot = _row_to_snapshot(row)
            if snapshot is None:
                continue
            if snapshot.signal_id in latest_by_signal_id:
                continue
            latest_by_signal_id[snapshot.signal_id] = snapshot
            if len(latest_by_signal_id) == len(signal_ids):
                break
        return list(latest_by_signal_id.values())

    def list_evaluation_snapshots_for_signals(
        self,
        *,
        signal_ids: list[str],
    ) -> list[EvaluationSnapshot]:
        if not signal_ids:
            return []
        placeholders = ", ".join("?" for _ in signal_ids)
        rows = self.conn.execute(
            f"""
            SELECT evaluation_id, subject_id, asset, target_id, signal_id, prediction_value,
                   observation_value, signed_edge, absolute_error, input_source,
                   input_range_start, input_range_end,
                   funding_cost_bps, borrow_fee_bps, roll_cost_bps, financing_cost_bps,
                   contract_multiplier, contract_id, contract_family, quote_ccy,
                   collateral_ccy, roll_event_json,
                   observation_spec_id, observable_id, adapter_kind,
                   signal_name, created_at
            FROM evaluation_snapshots
            WHERE signal_id IN ({placeholders})
            ORDER BY evaluation_id ASC, signal_id ASC
            """,
            tuple(signal_ids),
        ).fetchall()
        return [_row_to_snapshot(row) for row in rows if row is not None]

    def delete_evaluation_snapshots_for_signals(self, *, signal_ids: list[str]) -> int:
        if not signal_ids:
            return 0
        placeholders = ", ".join("?" for _ in signal_ids)
        with self.conn:
            self.conn.execute(
                f"""
                DELETE FROM evaluation_snapshots
                WHERE signal_id IN ({placeholders})
                """,
                tuple(signal_ids),
            )
            return int(self.conn.execute("SELECT changes()").fetchone()[0])

    def delete_non_latest_evaluation_snapshots_for_signals(
        self,
        *,
        signal_ids: list[str],
    ) -> int:
        if not signal_ids:
            return 0
        latest_snapshots = self.list_latest_evaluation_snapshots(signal_ids=signal_ids)
        latest_evaluation_id_by_signal_id = {
            item.signal_id: item.evaluation_id for item in latest_snapshots
        }
        deleted = 0
        with self.conn:
            for signal_id, evaluation_id in latest_evaluation_id_by_signal_id.items():
                self.conn.execute(
                    """
                    DELETE FROM evaluation_snapshots
                    WHERE signal_id = ?
                      AND evaluation_id != ?
                    """,
                    (signal_id, evaluation_id),
                )
                deleted += int(self.conn.execute("SELECT changes()").fetchone()[0])
        return deleted

    def archive_signal_discovery_run_evaluation_snapshots(
        self,
        *,
        signal_discovery_run_id: str,
        signal_ids: list[str],
    ) -> int:
        if not signal_ids:
            return 0
        placeholders = ", ".join("?" for _ in signal_ids)
        params: tuple[object, ...] = (signal_discovery_run_id, *signal_ids)
        with self.conn:
            self.conn.execute(
                f"""
                INSERT OR REPLACE INTO signal_discovery_run_evaluation_snapshots (
                    signal_discovery_run_id, evaluation_id, subject_id, asset, target_id,
                    signal_id, prediction_value, observation_value,
                    signed_edge, absolute_error, input_source,
                    input_range_start, input_range_end,
                    funding_cost_bps, borrow_fee_bps, roll_cost_bps, financing_cost_bps,
                    contract_multiplier, contract_id, contract_family, quote_ccy,
                    collateral_ccy, roll_event_json,
                    observation_spec_id, observable_id, adapter_kind,
                    signal_name, created_at
                )
                SELECT ?, evaluation_id, subject_id, asset, target_id,
                       signal_id, prediction_value, observation_value,
                       signed_edge, absolute_error, input_source,
                       input_range_start, input_range_end,
                       funding_cost_bps, borrow_fee_bps, roll_cost_bps, financing_cost_bps,
                       contract_multiplier, contract_id, contract_family, quote_ccy,
                       collateral_ccy, roll_event_json,
                       observation_spec_id, observable_id, adapter_kind,
                       signal_name, created_at
                FROM evaluation_snapshots
                WHERE signal_id IN ({placeholders})
                """,
                params,
            )
            return int(self.conn.execute("SELECT changes()").fetchone()[0])

    def list_signal_discovery_run_evaluation_snapshots(
        self,
        *,
        signal_discovery_run_id: str,
        signal_ids: list[str] | None = None,
    ) -> list[EvaluationSnapshot]:
        params: list[object] = [signal_discovery_run_id]
        where_clause = "WHERE signal_discovery_run_id = ?"
        if signal_ids is not None:
            if not signal_ids:
                return []
            placeholders = ", ".join("?" for _ in signal_ids)
            where_clause += f" AND signal_id IN ({placeholders})"
            params.extend(signal_ids)
        rows = self.conn.execute(
            f"""
            SELECT evaluation_id, subject_id, asset, target_id, signal_id, prediction_value,
                   observation_value, signed_edge, absolute_error, input_source,
                   input_range_start, input_range_end,
                   funding_cost_bps, borrow_fee_bps, roll_cost_bps, financing_cost_bps,
                   contract_multiplier, contract_id, contract_family, quote_ccy,
                   collateral_ccy, roll_event_json,
                   observation_spec_id, observable_id, adapter_kind,
                   signal_name, created_at
            FROM signal_discovery_run_evaluation_snapshots
            {where_clause}
            ORDER BY evaluation_id ASC, signal_id ASC
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_snapshot(row) for row in rows if row is not None]

    def record_prediction(
        self,
        *,
        evaluation_id: str,
        signal_id: str,
        prediction_value: float,
        recorded_at: str | None = None,
        subject_id: str | None = None,
        asset: str | None = None,
        target_id: str = DEFAULT_TARGET,
    ) -> tuple[PredictionRecord, bool]:
        self.ensure_schema()
        signal = self.get_signal(signal_id)
        if signal is None:
            raise ValueError(
                f"signal must exist before recording predictions: {signal_id}"
            )
        if signal.status != "active":
            raise ValueError(
                "prediction cannot be recorded while signal is "
                f"{signal.status}: {signal_id}"
            )
        resolved_asset = signal.asset if asset is None else asset
        if signal.asset != resolved_asset:
            raise ValueError(
                "prediction asset does not match signal asset: "
                f"{resolved_asset} != {signal.asset}"
            )
        resolved_subject_id = signal.subject_id if subject_id is None else subject_id
        if signal.subject_id != resolved_subject_id:
            raise ValueError(
                "prediction subject does not match signal subject: "
                f"{resolved_subject_id} != {signal.subject_id}"
            )
        if signal.target_id != target_id:
            raise ValueError(
                "prediction target does not match signal target: "
                f"{target_id} != {signal.target_id}"
            )
        self.register_target(target_id, recorded_at=recorded_at)

        existing = self.get_prediction(evaluation_id, signal_id)
        if existing is not None:
            if existing.value != float(prediction_value):
                raise ValueError(
                    "prediction already exists for this evaluation_id and signal_id with a "
                    "different value"
                )
            if (
                existing.subject_id != resolved_subject_id
                or existing.asset != resolved_asset
                or existing.target_id != target_id
            ):
                raise ValueError(
                    "prediction already exists for this evaluation with different subject/asset/target"
                )
            return existing, False

        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO predictions (
                    evaluation_id, signal_id, subject_id, asset, target_id, value, recorded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    signal_id,
                    resolved_subject_id,
                    resolved_asset,
                    target_id,
                    float(prediction_value),
                    timestamp,
                ),
            )

        prediction = self.get_prediction(evaluation_id, signal_id)
        assert prediction is not None
        return prediction, True

    def finalize_observation(
        self,
        *,
        evaluation_id: str,
        observation_value: float,
        recorded_at: str | None = None,
        subject_id: str | None = None,
        asset: str | None = None,
        target_id: str = DEFAULT_TARGET,
    ) -> tuple[ObservationRecord, bool]:
        self.ensure_schema()
        self.register_target(target_id, recorded_at=recorded_at)
        existing = self.get_observation(evaluation_id)
        resolved_subject_id = DEFAULT_SUBJECT_ID if subject_id is None else subject_id
        resolved_asset = (
            default_runtime_asset(resolved_subject_id)
            if asset is None
            else asset
        )
        if existing is not None:
            if existing.value != float(observation_value):
                raise ValueError(
                    "observation already exists for this evaluation_id with a different value"
                )
            if (
                existing.subject_id != resolved_subject_id
                or existing.asset != resolved_asset
                or existing.target_id != target_id
            ):
                raise ValueError(
                    "observation already exists for this evaluation with different subject/asset/target"
                )
            return existing, False

        timestamp = recorded_at or _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO observations (
                    evaluation_id, subject_id, asset, target_id, value, recorded_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    resolved_subject_id,
                    resolved_asset,
                    target_id,
                    float(observation_value),
                    timestamp,
                ),
            )

        observation = self.get_observation(evaluation_id)
        assert observation is not None
        return observation, True
