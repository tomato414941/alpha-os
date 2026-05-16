from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .contract_boundaries import active_constraint_stages, subject_set_contract_groups
from .data_repositories import EvaluationInputRepository, FeaturePlaneRepository
from .strategy_backtest import (
    run_strategy_backtest_from_store,
    subject_backtest_inputs_from_subject_set_planes,
)
from .evaluation_generation import generate_evaluation_inputs_batch_from_feature_plane
from .evaluation_inputs import EvaluationInput
from .evaluation_metric_config import DECISION_EVALUATION_METRIC_GROUP_NAMES
from .strategy_engine import (
    StrategyEvaluationInputRefs,
    StrategyEvaluationRequest,
)
from .evaluation_spec import EvaluationSpec
from .portfolio_construction_config import PortfolioConstructionSpec
from .evaluation_report import (
    EvaluationTaskResult,
    EvaluationMetricGroupResult,
    EvaluationFailureFindingGroup,
)
from .evaluation_report_repository import PendingEvaluationDecisionTrace
from .evaluation_report_service import build_report_evaluation_task_contract_fields
from .portfolio_decision import SubjectSet
from .signal_discovery_strategy_evaluation import (
    build_signal_discovery_strategy_evaluation_metric_group_results,
)
from .signal_registry import SignalDefinition
from .store import EvaluationSnapshot, _utc_now
from .strategy_sleeves import SleeveAttributionSummary, StrategySleeveCompositionSpec
from .subject_set_feature_plane import SubjectPlaneKey, build_subject_set_feature_planes
from .subject_set_facts import format_subject_set_facts
from .trading_strategy import TradingStrategySpec
from .universe_contract import validate_subject_set_universe_contract


class EvaluationExecutionReadPort(Protocol):
    def get_trading_strategy(self, strategy_id: str):
        ...

    def get_subject_set(self, subject_set_id: str):
        ...

    def get_strategy_checkpoint(self, strategy_checkpoint_id: str):
        ...

    def get_screening_result(self, screening_result_id: str):
        ...

    def get_compressed_belief(self, compressed_belief_id: str):
        ...

    def get_signal(self, signal_id: str):
        ...

    def list_prepared_evaluation_snapshots(
        self,
        *,
        snapshot_set_id: str,
        signal_ids: list[str] | None = None,
    ) -> list[EvaluationSnapshot]:
        ...

    def list_evaluation_snapshots_for_signals(
        self,
        *,
        signal_ids: list[str],
    ) -> list[EvaluationSnapshot]:
        ...


@dataclass(frozen=True)
class EvaluationExecutionContext:
    store: EvaluationExecutionReadPort
    evaluation_spec: EvaluationSpec
    feature_plane_repository: FeaturePlaneRepository | None = None
    evaluation_input_repository: EvaluationInputRepository | None = None


@dataclass(frozen=True)
class EvaluationExecutionResult:
    task_result: EvaluationTaskResult
    pending_decision_traces: tuple[PendingEvaluationDecisionTrace, ...] = ()


@dataclass(frozen=True)
class PreparedStrategyEvaluationInputs:
    strategy_checkpoint: object | None
    snapshot_set_id: str | None
    prepared_start_date: str
    prepared_end_date: str
    screening_state: object
    compressed_belief_state: object


class EvaluationExecutionStrategy(Protocol):
    def run(
        self,
        *,
        execution_request: StrategyEvaluationRequest,
        context: EvaluationExecutionContext,
    ) -> EvaluationExecutionResult: ...


def resolve_prepared_strategy_evaluation_inputs(
    *,
    store: EvaluationExecutionReadPort,
    input_refs: StrategyEvaluationInputRefs,
) -> PreparedStrategyEvaluationInputs:
    strategy_checkpoint_record = store.get_strategy_checkpoint(input_refs.strategy_checkpoint_id)
    if strategy_checkpoint_record is None:
        raise ValueError(
            f"strategy checkpoint does not exist: {input_refs.strategy_checkpoint_id}"
        )
    strategy_checkpoint = strategy_checkpoint_record.state
    screening_state = store.get_screening_result(strategy_checkpoint.screening_result_id)
    if screening_state is None:
        raise ValueError(
            f"screening result does not exist: {strategy_checkpoint.screening_result_id}"
        )
    compressed_belief_state = store.get_compressed_belief(
        strategy_checkpoint.compressed_belief_id
    )
    if compressed_belief_state is None:
        raise ValueError(
            f"compressed belief does not exist: {strategy_checkpoint.compressed_belief_id}"
        )
    return PreparedStrategyEvaluationInputs(
        strategy_checkpoint=strategy_checkpoint,
        snapshot_set_id=strategy_checkpoint.snapshot_set_id,
        prepared_start_date=strategy_checkpoint.execution_start_date,
        prepared_end_date=strategy_checkpoint.execution_end_date,
        screening_state=screening_state,
        compressed_belief_state=compressed_belief_state,
    )


def resolve_prepared_strategy_survivor_snapshots(
    *,
    execution_request: StrategyEvaluationRequest,
    context: EvaluationExecutionContext,
    prepared_inputs: PreparedStrategyEvaluationInputs,
) -> list[EvaluationSnapshot]:
    store = context.store
    strategy_checkpoint = prepared_inputs.strategy_checkpoint
    snapshot_set_id = prepared_inputs.snapshot_set_id
    screening_state = prepared_inputs.screening_state
    survivor_signal_ids = (
        list(strategy_checkpoint.survivor_signal_ids)
        if strategy_checkpoint is not None
        else [item.signal_id for item in screening_state.result.survivors]
    )
    if requires_frozen_test_application(
        prepared_start_date=prepared_inputs.prepared_start_date,
        prepared_end_date=prepared_inputs.prepared_end_date,
        evaluation_date_ranges=execution_request.evaluation_date_ranges,
    ):
        if not execution_request.context.base_url:
            raise ValueError("evaluation task is missing base_url")
        frozen_definitions = frozen_survivor_definitions(
            store,
            signal_ids=survivor_signal_ids,
        )
        return generate_frozen_survivor_test_snapshots(
            store,
            subject_set_id=execution_request.context.subject_set_id,
            survivor_signal_ids=survivor_signal_ids,
            start_date=frozen_snapshot_start_date(
                evaluation_date_ranges=execution_request.evaluation_date_ranges,
                executable_definitions=frozen_definitions,
                metric_window=max(context.evaluation_spec.metric_windows),
                portfolio_construction=execution_request.context.portfolio_construction,
                trading_calendar=_trading_calendar_for_subject_set(
                    store,
                    execution_request.context.subject_set_id,
                ),
            ),
            end_date=max(item.end_date for item in execution_request.evaluation_date_ranges),
            base_url=execution_request.context.base_url,
            feature_plane_repository=context.feature_plane_repository,
            evaluation_input_repository=context.evaluation_input_repository,
        )
    survivor_snapshots = []
    if snapshot_set_id is not None:
        survivor_snapshots = store.list_prepared_evaluation_snapshots(
            snapshot_set_id=snapshot_set_id,
            signal_ids=survivor_signal_ids,
        )
    if not survivor_snapshots:
        survivor_snapshots = store.list_evaluation_snapshots_for_signals(
            signal_ids=survivor_signal_ids
        )
    return survivor_snapshots


def _subject_metadata_by_subject(
    subject_set: SubjectSet | None,
) -> dict[str, dict[str, str]]:
    if subject_set is None:
        return {}
    metadata: dict[str, dict[str, str]] = {}
    for subject_id in subject_set.subject_ids:
        instrument = subject_set.instrument_for_subject(subject_id)
        if instrument is None:
            metadata[subject_id] = {}
            continue
        values = {
            "asset_class": instrument.asset_class,
            "cluster": instrument.cluster,
        }
        metadata[subject_id] = {key: value for key, value in values.items() if value is not None}
    return metadata


def _constraint_stages_for_entry(execution_request: StrategyEvaluationRequest):
    return active_constraint_stages(
        execution_request.context.portfolio_construction.constraint_boundary,
        field_values={
            "direction_mode": (
                execution_request.context.portfolio_construction.direction_mode
                if execution_request.context.portfolio_construction.direction_mode
                != "long_short"
                else None
            ),
            "gross_exposure_cap": execution_request.context.portfolio_construction.gross_exposure_cap,
            "target_vol": execution_request.context.portfolio_construction.target_vol,
            "gross_leverage_cap": execution_request.context.portfolio_construction.gross_leverage_cap,
            "net_exposure_target": execution_request.context.portfolio_construction.net_exposure_target,
            "asset_class_weight_caps": (execution_request.context.portfolio_construction.asset_class_weight_caps),
            "cluster_weight_caps": execution_request.context.portfolio_construction.cluster_weight_caps,
        },
    )


def _is_range_within_execution_range(
    *,
    execution_start_date: str,
    execution_end_date: str,
    evaluation_date_range,
) -> bool:
    return (
        execution_start_date <= evaluation_date_range.start_date
        and evaluation_date_range.end_date <= execution_end_date
    )


def requires_frozen_test_application(
    *,
    prepared_start_date: str,
    prepared_end_date: str,
    evaluation_date_ranges,
) -> bool:
    return any(
        not _is_range_within_execution_range(
            execution_start_date=prepared_start_date,
            execution_end_date=prepared_end_date,
            evaluation_date_range=item,
        )
        for item in evaluation_date_ranges
    )


def frozen_survivor_definitions(
    store: EvaluationExecutionReadPort,
    *,
    signal_ids: list[str],
) -> list[SignalDefinition]:
    definitions: list[SignalDefinition] = []
    for signal_id in signal_ids:
        state = store.get_signal(signal_id)
        if state is None:
            raise ValueError(f"frozen survivor signal does not exist: {signal_id}")
        if state.definition is None:
            raise ValueError(f"frozen survivor signal is missing definition: {signal_id}")
        definitions.append(
            SignalDefinition.from_document(
                signal_id=state.signal_id,
                document=state.definition,
                asset=state.asset,
            )
        )
    return definitions


def frozen_snapshot_start_date(
    *,
    evaluation_date_ranges,
    executable_definitions: list[SignalDefinition],
    metric_window: int,
    portfolio_construction: PortfolioConstructionSpec,
    trading_calendar: str | None = None,
) -> str:
    evaluation_start = min(item.start_date for item in evaluation_date_ranges)
    max_lookback = max((item.lookback for item in executable_definitions), default=1)
    warmup_steps = max(max_lookback, max(int(metric_window), 1))
    if portfolio_construction.sizing_method in {
        "signed_mean_variance",
        "equal_weight",
        "minimum_variance",
        "risk_budgeting",
        "hierarchical_risk_parity",
        "conviction_adjusted_hierarchical_risk_parity",
    }:
        warmup_steps = max(warmup_steps, 20)
    offset = (
        pd.Timedelta(days=warmup_steps)
        if trading_calendar in {"fixture_daily", "crypto_daily", "daily", "24x7"}
        else pd.offsets.BDay(warmup_steps)
    )
    return str((pd.Timestamp(evaluation_start) - offset).date())


def _trading_calendar_for_subject_set(
    store: EvaluationExecutionReadPort,
    subject_set_id: str,
) -> str | None:
    subject_set_state = store.get_subject_set(subject_set_id)
    if subject_set_state is None:
        return None
    return subject_set_state.definition.universe_policy.trading_calendar


def generate_frozen_survivor_test_snapshots(
    store: EvaluationExecutionReadPort,
    *,
    subject_set_id: str,
    survivor_signal_ids: list[str],
    start_date: str,
    end_date: str,
    base_url: str,
    feature_plane_repository: FeaturePlaneRepository | None = None,
    evaluation_input_repository: EvaluationInputRepository | None = None,
) -> list[EvaluationSnapshot]:
    subject_set_state = store.get_subject_set(subject_set_id)
    if subject_set_state is None:
        raise ValueError(f"subject set does not exist: {subject_set_id}")
    validate_subject_set_universe_contract(subject_set_state.definition)
    executable_definitions = frozen_survivor_definitions(
        store,
        signal_ids=survivor_signal_ids,
    )
    if not executable_definitions:
        return []
    subject_planes = build_subject_set_feature_planes(
        subject_set=subject_set_state.definition,
        executable_definitions=executable_definitions,
        base_url=base_url,
        feature_plane_repository=feature_plane_repository,
    )
    grouped_definitions: dict[tuple[str, str], list[SignalDefinition]] = {}
    for definition in executable_definitions:
        if definition.observation_spec is None:
            raise ValueError(f"frozen survivor is missing observation spec: {definition.signal_id}")
        grouped_definitions.setdefault(
            (
                definition.asset,
                definition.observation_spec.observation_spec_id,
            ),
            [],
        ).append(definition)

    evaluation_inputs: list[EvaluationInput] = []
    for (asset, observation_spec_id), definitions in grouped_definitions.items():
        plane = subject_planes.get(
            SubjectPlaneKey(
                asset=asset,
                observation_spec_id=observation_spec_id,
            )
        )
        if plane is None:
            raise ValueError(
                f"frozen survivor feature plane is missing: {asset}/{observation_spec_id}"
            )
        binding = next(
            (
                item
                for item in subject_set_state.definition.bindings
                if item.asset == asset and item.observation_spec_id == observation_spec_id
            ),
            None,
        )
        contract_multiplier = None
        contract_family = None
        quote_ccy = None
        collateral_ccy = None
        roll_rule = None
        if binding is not None:
            instrument = subject_set_state.definition.instrument_for_subject(binding.subject_id)
            if instrument is not None:
                if instrument.multiplier is not None:
                    contract_multiplier = float(instrument.multiplier)
                contract_family = instrument.contract_family
                quote_ccy = instrument.quote_ccy
                collateral_ccy = instrument.collateral_ccy
                roll_rule = instrument.roll_rule
        if evaluation_input_repository is not None:
            evaluation_inputs.extend(
                evaluation_input_repository.load_inputs_for_range(
                    plane=plane,
                    definitions=definitions,
                    start_date=start_date,
                    end_date=end_date,
                    observation_spec=definitions[0].observation_spec,
                    asset=asset,
                    base_url=base_url,
                    contract_multiplier=contract_multiplier,
                    contract_family=contract_family,
                    quote_ccy=quote_ccy,
                    collateral_ccy=collateral_ccy,
                    roll_rule=roll_rule,
                )
            )
        else:
            evaluation_inputs.extend(
                generate_evaluation_inputs_batch_from_feature_plane(
                    plane=plane,
                    start_date=start_date,
                    end_date=end_date,
                    definitions=definitions,
                    observation_spec=definitions[0].observation_spec,
                    contract_multiplier=contract_multiplier,
                    contract_family=contract_family,
                    quote_ccy=quote_ccy,
                    collateral_ccy=collateral_ccy,
                    roll_rule=roll_rule,
                )
            )

    definition_by_signal_id = {item.signal_id: item for item in executable_definitions}
    created_at = _utc_now()
    snapshots = []
    for item in evaluation_inputs:
        definition = definition_by_signal_id[item.signal_id]
        observation_spec = definition.observation_spec
        evaluation_id = item.evaluation_id or f"{item.subject_id}:{item.target_id}:{item.date}"
        snapshots.append(
            EvaluationSnapshot(
                evaluation_id=evaluation_id,
                subject_id=item.subject_id,
                asset=definition.asset,
                target_id=item.target_id,
                signal_id=item.signal_id,
                prediction_value=float(item.prediction),
                observation_value=float(item.observation),
                signed_edge=float(item.prediction * item.observation),
                absolute_error=float(abs(item.prediction - item.observation)),
                input_source="walk_forward_test_application",
                input_range_start=start_date,
                input_range_end=end_date,
                funding_cost_bps=item.funding_cost_bps,
                borrow_fee_bps=item.borrow_fee_bps,
                roll_cost_bps=item.roll_cost_bps,
                contract_multiplier=item.contract_multiplier,
                observation_spec_id=(
                    None if observation_spec is None else observation_spec.observation_spec_id
                ),
                observable_id=(
                    None if observation_spec is None else observation_spec.observable_id
                ),
                adapter_kind=(None if observation_spec is None else observation_spec.adapter_kind),
                created_at=created_at,
            )
        )
    return snapshots


def strategy_sleeve_attribution_summaries(
    trading_strategy: TradingStrategySpec | None,
    subject_set: SubjectSet | None,
    *,
    sleeve_composition: StrategySleeveCompositionSpec | None = None,
) -> tuple[SleeveAttributionSummary, ...]:
    composition = sleeve_composition
    if composition is None and trading_strategy is not None:
        composition = trading_strategy.portfolio.portfolio_construction.sleeve_composition
    if composition is None:
        return ()
    subject_ids = () if subject_set is None else subject_set.subject_ids
    summaries: list[SleeveAttributionSummary] = []
    for sleeve in composition.enabled_sleeves:
        eligible_subject_ids = set(subject_ids)
        subject_filter = sleeve.subject_filter
        if subject_filter.subject_ids:
            eligible_subject_ids &= set(subject_filter.subject_ids)
        if subject_set is not None:
            eligible_subject_ids = {
                subject_id
                for subject_id in eligible_subject_ids
                if subject_matches_sleeve_filter(
                    subject_set,
                    subject_id=subject_id,
                    instrument_types=subject_filter.instrument_types,
                    asset_classes=subject_filter.asset_classes,
                    regions=subject_filter.regions,
                    clusters=subject_filter.clusters,
                )
            }
        summaries.append(
            SleeveAttributionSummary(
                sleeve_id=sleeve.sleeve_id,
                sleeve_kind=sleeve.sleeve_kind,
                risk_budget=sleeve.risk_budget,
                subject_count=len(eligible_subject_ids),
            )
        )
    return tuple(summaries)


def subject_matches_sleeve_filter(
    subject_set: SubjectSet,
    *,
    subject_id: str,
    instrument_types: tuple[str, ...],
    asset_classes: tuple[str, ...],
    regions: tuple[str, ...],
    clusters: tuple[str, ...],
) -> bool:
    instrument = subject_set.instrument_for_subject(subject_id)
    if instrument is None:
        return not any((instrument_types, asset_classes, regions, clusters))
    checks = (
        (instrument.instrument_type, instrument_types),
        (instrument.asset_class, asset_classes),
        (instrument.region, regions),
        (instrument.cluster, clusters),
    )
    return all(not allowed_values or value in allowed_values for value, allowed_values in checks)


@dataclass(frozen=True)
class DirectStrategyEvaluationExecutionStrategy:
    def run(
        self,
        *,
        execution_request: StrategyEvaluationRequest,
        context: EvaluationExecutionContext,
    ) -> EvaluationExecutionResult:
        store = context.store
        strategy_state = store.get_trading_strategy(execution_request.context.strategy_id)
        subject_set_state = store.get_subject_set(execution_request.context.subject_set_id)
        if subject_set_state is not None:
            validate_subject_set_universe_contract(subject_set_state.definition)
        direct_evaluation = run_strategy_backtest_from_store(
            store=store,
            strategy_id=execution_request.context.strategy_id,
            subject_set_id=execution_request.context.subject_set_id,
            target_id=execution_request.context.target_id,
            evaluation_date_ranges=execution_request.evaluation_date_ranges,
            base_url=execution_request.context.base_url,
            portfolio_construction=execution_request.context.portfolio_construction,
            rebalance_friction_policy=execution_request.context.rebalance_friction_policy,
            execution_cost_assumptions=execution_request.context.execution_cost_assumptions,
            holding_cost_assumptions=execution_request.context.holding_cost_assumptions,
            feature_plane_repository=context.feature_plane_repository,
        )
        direct_metric_group_results, direct_failure_finding_groups = direct_evaluation
        subject_set = None if subject_set_state is None else subject_set_state.definition
        pending_traces = tuple(
            PendingEvaluationDecisionTrace(
                evaluation_task_id=execution_request.evaluation_task_id,
                evaluation_fold_label=execution_request.fold_label,
                evaluation_range_label=trace_result.range_label,
                result=trace_result.result,
                subject_metadata_by_subject=_subject_metadata_by_subject(subject_set),
            )
            for trace_result in getattr(direct_evaluation, "selected_trace_results", ())
        )
        return EvaluationExecutionResult(
            task_result=EvaluationTaskResult(
                evaluation_task_id=execution_request.evaluation_task_id,
                construction_kind=execution_request.context.portfolio_construction.construction_kind,
                strategy_id=execution_request.context.strategy_id,
                strategy_contract_fields=build_report_evaluation_task_contract_fields(
                    execution_request.context.portfolio_construction,
                    rebalance_friction_policy=execution_request.context.rebalance_friction_policy,
                    execution_cost_assumptions=execution_request.context.execution_cost_assumptions,
                    holding_cost_assumptions=execution_request.context.holding_cost_assumptions,
                    subject_set=subject_set,
                    subject_set_id=execution_request.context.subject_set_id,
                    target_id=execution_request.context.target_id,
                    selection_kind=execution_request.context.selection_kind,
                    top_k=execution_request.context.top_k,
                ),
                subject_set_facts=(
                    None if subject_set is None else format_subject_set_facts(subject_set)
                ),
                subject_set_contract_groups=(
                    ()
                    if subject_set is None
                    else subject_set_contract_groups(subject_set.contract_boundary)
                ),
                universe_policy_fields=(
                    {} if subject_set is None else subject_set.universe_policy.to_document()
                ),
                constraint_stages=_constraint_stages_for_entry(execution_request),
                sleeve_attribution_summaries=strategy_sleeve_attribution_summaries(
                    None if strategy_state is None else strategy_state.trading_strategy,
                    subject_set,
                    sleeve_composition=execution_request.context.portfolio_construction.sleeve_composition,
                ),
                metric_group_results=tuple(
                    direct_metric_group_results[metric_group_name]
                    for metric_group_name in execution_request.metric_group_names
                    if metric_group_name in direct_metric_group_results
                ),
                failure_finding_groups=direct_failure_finding_groups,
                artifact_refs={
                    "evaluation_task_ids": (execution_request.evaluation_task_id,),
                    "strategy_ids": (execution_request.context.strategy_id,),
                    "evaluation_fold_labels": (execution_request.fold_label,),
                    "evaluation_range_labels": tuple(
                        item.label for item in execution_request.evaluation_date_ranges
                    ),
                },
            ),
            pending_decision_traces=pending_traces,
        )


@dataclass(frozen=True)
class PreparedStrategyEvaluationExecutionStrategy:
    def run(
        self,
        *,
        execution_request: StrategyEvaluationRequest,
        context: EvaluationExecutionContext,
    ) -> EvaluationExecutionResult:
        store = context.store
        input_refs = execution_request.input_refs
        if input_refs is None:
            raise ValueError("prepared strategy evaluation requires input refs")
        prepared_inputs = resolve_prepared_strategy_evaluation_inputs(
            store=store,
            input_refs=input_refs,
        )
        metric_group_result_map: dict[str, EvaluationMetricGroupResult] = {}
        strategy_checkpoint = prepared_inputs.strategy_checkpoint
        screening_state = prepared_inputs.screening_state
        compressed_belief_state = prepared_inputs.compressed_belief_state
        artifact_refs: dict[str, tuple[str, ...]] = {
            "evaluation_task_ids": (execution_request.evaluation_task_id,),
            "strategy_ids": (execution_request.context.strategy_id,),
            "screening_result_ids": (screening_state.screening_result_id,),
            "compressed_belief_ids": (compressed_belief_state.compressed_belief_id,),
            "evaluation_fold_labels": (execution_request.fold_label,),
        }
        if strategy_checkpoint is not None:
            artifact_refs["strategy_checkpoint_ids"] = (
                strategy_checkpoint.strategy_checkpoint_id,
            )
        failure_finding_groups: tuple[EvaluationFailureFindingGroup, ...] = ()
        subject_set = None
        pending_decision_traces: tuple[PendingEvaluationDecisionTrace, ...] = ()
        decision_metric_group_names = tuple(
            item
            for item in execution_request.metric_group_names
            if item in DECISION_EVALUATION_METRIC_GROUP_NAMES
        )
        if decision_metric_group_names:
            (
                subject_set,
                metric_group_result_map,
                failure_finding_groups,
                pending_decision_traces,
            ) = self._run_decision_evaluation_results(
                execution_request=execution_request,
                context=context,
                prepared_inputs=prepared_inputs,
                metric_group_result_map=metric_group_result_map,
            )
            artifact_refs["evaluation_range_labels"] = tuple(
                item.label for item in execution_request.evaluation_date_ranges
            )

        strategy_state = store.get_trading_strategy(execution_request.context.strategy_id)
        signal_discovery_id = (
            None if strategy_checkpoint is None else strategy_checkpoint.signal_discovery_id
        )
        return EvaluationExecutionResult(
            task_result=EvaluationTaskResult(
                evaluation_task_id=execution_request.evaluation_task_id,
                construction_kind=execution_request.context.portfolio_construction.construction_kind,
                strategy_id=execution_request.context.strategy_id,
                signal_discovery_id=signal_discovery_id,
                strategy_contract_fields=build_report_evaluation_task_contract_fields(
                    execution_request.context.portfolio_construction,
                    rebalance_friction_policy=execution_request.context.rebalance_friction_policy,
                    execution_cost_assumptions=execution_request.context.execution_cost_assumptions,
                    holding_cost_assumptions=execution_request.context.holding_cost_assumptions,
                    subject_set=subject_set,
                    subject_set_id=execution_request.context.subject_set_id,
                    target_id=execution_request.context.target_id,
                    selection_kind=execution_request.context.selection_kind,
                    top_k=execution_request.context.top_k,
                ),
                subject_set_facts=(
                    None if subject_set is None else format_subject_set_facts(subject_set)
                ),
                subject_set_contract_groups=(
                    ()
                    if subject_set is None
                    else subject_set_contract_groups(subject_set.contract_boundary)
                ),
                universe_policy_fields=(
                    {} if subject_set is None else subject_set.universe_policy.to_document()
                ),
                constraint_stages=_constraint_stages_for_entry(execution_request),
                sleeve_attribution_summaries=strategy_sleeve_attribution_summaries(
                    None if strategy_state is None else strategy_state.trading_strategy,
                    subject_set,
                    sleeve_composition=execution_request.context.portfolio_construction.sleeve_composition,
                ),
                metric_group_results=tuple(
                    metric_group_result_map[metric_group_name]
                    for metric_group_name in execution_request.metric_group_names
                    if metric_group_name in metric_group_result_map
                ),
                failure_finding_groups=failure_finding_groups,
                artifact_refs=artifact_refs,
            ),
            pending_decision_traces=pending_decision_traces,
        )

    def _run_decision_evaluation_results(
        self,
        *,
        execution_request: StrategyEvaluationRequest,
        context: EvaluationExecutionContext,
        prepared_inputs: PreparedStrategyEvaluationInputs,
        metric_group_result_map: dict[str, EvaluationMetricGroupResult],
    ):
        store = context.store
        protocol = context.evaluation_spec
        screening_state = prepared_inputs.screening_state
        compressed_belief_state = prepared_inputs.compressed_belief_state
        survivor_snapshots = resolve_prepared_strategy_survivor_snapshots(
            execution_request=execution_request,
            context=context,
            prepared_inputs=prepared_inputs,
        )
        subject_set_state = store.get_subject_set(execution_request.context.subject_set_id)
        if subject_set_state is not None:
            validate_subject_set_universe_contract(subject_set_state.definition)
        subject_set = None if subject_set_state is None else subject_set_state.definition
        funding_cost_bps_series_by_subject: dict[str, pd.Series] = {}
        borrow_fee_bps_series_by_subject: dict[str, pd.Series] = {}
        roll_cost_bps_series_by_subject: dict[str, pd.Series] = {}
        contract_multiplier_by_subject: dict[str, float] = {}
        snapshots_missing_artifacts = any(
            item.funding_cost_bps is None
            or item.borrow_fee_bps is None
            or item.roll_cost_bps is None
            or item.contract_multiplier is None
            for item in survivor_snapshots
        )
        if subject_set is not None and execution_request.context.base_url and snapshots_missing_artifacts:
            subject_planes = build_subject_set_feature_planes(
                subject_set=subject_set,
                executable_definitions=[],
                base_url=execution_request.context.base_url,
                feature_plane_repository=context.feature_plane_repository,
            )
            (
                _,
                _,
                funding_cost_bps_series_by_subject,
                borrow_fee_bps_series_by_subject,
                roll_cost_bps_series_by_subject,
                contract_multiplier_by_subject,
            ) = subject_backtest_inputs_from_subject_set_planes(
                subject_set=subject_set,
                subject_planes=subject_planes,
            )
        native_evaluation = build_signal_discovery_strategy_evaluation_metric_group_results(
            screening_result=screening_state.result,
            compressed_belief=compressed_belief_state.belief,
            subject_set_id=execution_request.context.subject_set_id,
            subject_set=subject_set,
            funding_cost_bps_series_by_subject=funding_cost_bps_series_by_subject,
            borrow_fee_bps_series_by_subject=borrow_fee_bps_series_by_subject,
            roll_cost_bps_series_by_subject=roll_cost_bps_series_by_subject,
            contract_multiplier_by_subject=contract_multiplier_by_subject,
            target_id=execution_request.context.target_id,
            snapshots=survivor_snapshots,
            evaluation_date_ranges=execution_request.evaluation_date_ranges,
            metric_window=max(protocol.metric_windows),
            portfolio_construction=execution_request.context.portfolio_construction,
            rebalance_friction_policy=execution_request.context.rebalance_friction_policy,
            execution_cost_assumptions=execution_request.context.execution_cost_assumptions,
            holding_cost_assumptions=execution_request.context.holding_cost_assumptions,
            top_k=execution_request.context.top_k,
        )
        native_metric_group_results, failure_finding_groups = native_evaluation
        subject_metadata_by_subject = _subject_metadata_by_subject(subject_set)
        pending_decision_traces = tuple(
            PendingEvaluationDecisionTrace(
                evaluation_task_id=execution_request.evaluation_task_id,
                evaluation_fold_label=execution_request.fold_label,
                evaluation_range_label=trace_result.range_label,
                result=trace_result.result,
                subject_metadata_by_subject=subject_metadata_by_subject,
            )
            for trace_result in getattr(native_evaluation, "selected_trace_results", ())
        )
        metric_group_result_map.update(native_metric_group_results)
        return (
            subject_set,
            metric_group_result_map,
            failure_finding_groups,
            pending_decision_traces,
        )


def evaluation_execution_strategy_for_request(
    execution_request: StrategyEvaluationRequest,
) -> EvaluationExecutionStrategy:
    if execution_request.input_refs is None:
        return DirectStrategyEvaluationExecutionStrategy()
    return PreparedStrategyEvaluationExecutionStrategy()
