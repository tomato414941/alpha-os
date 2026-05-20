from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contract_boundaries import active_constraint_stages, subject_set_contract_groups
from .data_repositories import FeaturePlaneRepository
from .evaluation_cost_config import (
    EvaluationRebalanceFrictionPolicySpec,
    ExecutionCostAssumptionsSpec,
    HoldingCostAssumptionsSpec,
)
from .strategy_backtest import run_strategy_backtest_from_store
from .strategy_engine import StrategyEvaluationRequest
from .evaluation_spec import EvaluationSpec
from .portfolio_construction_config import PortfolioConstructionSpec
from .evaluation_result import EvaluationTaskResult
from .evaluation_task_contract_fields import build_evaluation_task_contract_fields
from .portfolio_decision import SubjectSet
from .strategy_sleeves import SleeveAttributionSummary, StrategySleeveCompositionSpec
from .subject_set_facts import format_subject_set_facts
from .trading_strategy import TradingStrategySpec
from .universe_contract import validate_subject_set_universe_contract


class EvaluationExecutionReadPort(Protocol):
    def get_trading_strategy(self, strategy_id: str):
        ...

    def get_subject_set(self, subject_set_id: str):
        ...


@dataclass(frozen=True)
class EvaluationExecutionContext:
    store: EvaluationExecutionReadPort
    evaluation_spec: EvaluationSpec
    feature_plane_repository: FeaturePlaneRepository | None = None


def _portfolio_construction_for_strategy(
    trading_strategy: TradingStrategySpec,
) -> PortfolioConstructionSpec:
    return trading_strategy.portfolio.portfolio_construction


def _rebalance_friction_policy_for_strategy(
    trading_strategy: TradingStrategySpec,
) -> EvaluationRebalanceFrictionPolicySpec:
    strategy_policy = trading_strategy.portfolio.rebalance_friction_policy
    if strategy_policy is not None:
        return EvaluationRebalanceFrictionPolicySpec.from_document(
            {
                key: value
                for key, value in strategy_policy.to_document().items()
                if value is not None
            }
        )
    raise ValueError(
        "trading strategy is missing rebalance_friction_policy: "
        f"{trading_strategy.strategy_id}"
    )


def _execution_cost_assumptions_for_strategy(
    trading_strategy: TradingStrategySpec,
) -> ExecutionCostAssumptionsSpec:
    strategy_policy = trading_strategy.portfolio.execution_policy
    if strategy_policy is not None:
        return ExecutionCostAssumptionsSpec(
            market_impact_bps=strategy_policy.market_impact_bps or 0.0,
            fee_bps=strategy_policy.fee_bps or 0.0,
            bid_ask_spread_bps=strategy_policy.bid_ask_spread_bps or 0.0,
        )
    raise ValueError(
        "trading strategy is missing execution_policy: "
        f"{trading_strategy.strategy_id}"
    )


def _holding_cost_assumptions_for_strategy(
    trading_strategy: TradingStrategySpec,
) -> HoldingCostAssumptionsSpec:
    strategy_policy = trading_strategy.portfolio.holding_cost_policy
    if strategy_policy is not None:
        return HoldingCostAssumptionsSpec(
            funding_bps_per_step=(
                0.0
                if strategy_policy.funding_bps_per_step is None
                else strategy_policy.funding_bps_per_step
            ),
            borrow_fee_bps_per_step=(
                0.0
                if strategy_policy.borrow_fee_bps_per_step is None
                else strategy_policy.borrow_fee_bps_per_step
            ),
        )
    raise ValueError(
        "trading strategy is missing holding_cost_policy: "
        f"{trading_strategy.strategy_id}"
    )


def _trading_strategy_for_request(
    store: EvaluationExecutionReadPort,
    execution_request: StrategyEvaluationRequest,
) -> TradingStrategySpec:
    strategy_state = store.get_trading_strategy(execution_request.context.strategy_id)
    if strategy_state is None:
        raise ValueError(
            "evaluation task strategy does not exist: "
            f"{execution_request.context.strategy_id}"
        )
    return strategy_state.trading_strategy


def _subject_set_id_for_strategy(trading_strategy: TradingStrategySpec) -> str:
    subject_set_id = trading_strategy.subject_set_id
    if not isinstance(subject_set_id, str) or not subject_set_id:
        raise ValueError(
            "evaluation task strategy is missing subject_set: "
            f"{trading_strategy.strategy_id}"
        )
    return subject_set_id


def _constraint_stages_for_portfolio_construction(
    portfolio_construction: PortfolioConstructionSpec,
):
    return active_constraint_stages(
        portfolio_construction.constraint_boundary,
        field_values={
            "direction_mode": (
                portfolio_construction.direction_mode
                if portfolio_construction.direction_mode
                != "long_short"
                else None
            ),
            "gross_exposure_cap": portfolio_construction.gross_exposure_cap,
            "target_vol": portfolio_construction.target_vol,
            "gross_leverage_cap": portfolio_construction.gross_leverage_cap,
            "net_exposure_target": portfolio_construction.net_exposure_target,
            "asset_class_weight_caps": portfolio_construction.asset_class_weight_caps,
            "cluster_weight_caps": portfolio_construction.cluster_weight_caps,
        },
    )


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


def run_strategy_evaluation_task(
    execution_request: StrategyEvaluationRequest,
    *,
    context: EvaluationExecutionContext,
) -> EvaluationTaskResult:
    store = context.store
    trading_strategy = _trading_strategy_for_request(store, execution_request)
    portfolio_construction = _portfolio_construction_for_strategy(trading_strategy)
    rebalance_friction_policy = _rebalance_friction_policy_for_strategy(
        trading_strategy
    )
    execution_cost_assumptions = _execution_cost_assumptions_for_strategy(
        trading_strategy
    )
    holding_cost_assumptions = _holding_cost_assumptions_for_strategy(
        trading_strategy
    )
    subject_set_id = _subject_set_id_for_strategy(trading_strategy)
    subject_set_state = store.get_subject_set(subject_set_id)
    if subject_set_state is not None:
        validate_subject_set_universe_contract(subject_set_state.definition)
    direct_evaluation = run_strategy_backtest_from_store(
        store=store,
        strategy_id=execution_request.context.strategy_id,
        subject_set_id=subject_set_id,
        target_id=execution_request.context.target_id,
        evaluation_date_ranges=execution_request.evaluation_date_ranges,
        base_url=execution_request.context.base_url,
        portfolio_construction=portfolio_construction,
        rebalance_friction_policy=rebalance_friction_policy,
        execution_cost_assumptions=execution_cost_assumptions,
        holding_cost_assumptions=holding_cost_assumptions,
        feature_plane_repository=context.feature_plane_repository,
    )
    direct_metric_group_results, direct_failure_finding_groups = direct_evaluation
    subject_set = None if subject_set_state is None else subject_set_state.definition
    return EvaluationTaskResult(
        evaluation_task_id=execution_request.evaluation_task_id,
        construction_kind=portfolio_construction.construction_kind,
        strategy_id=execution_request.context.strategy_id,
        strategy_contract_fields=build_evaluation_task_contract_fields(
            portfolio_construction,
            rebalance_friction_policy=rebalance_friction_policy,
            execution_cost_assumptions=execution_cost_assumptions,
            holding_cost_assumptions=holding_cost_assumptions,
            target_id=execution_request.context.target_id,
            selection_kind=trading_strategy.selection_kind,
            top_k=trading_strategy.portfolio.top_k,
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
        constraint_stages=_constraint_stages_for_portfolio_construction(
            portfolio_construction
        ),
        sleeve_attribution_summaries=strategy_sleeve_attribution_summaries(
            trading_strategy,
            subject_set,
            sleeve_composition=portfolio_construction.sleeve_composition,
        ),
        metric_group_results=tuple(
            direct_metric_group_results[metric_group_name]
            for metric_group_name in execution_request.metric_group_names
            if metric_group_name in direct_metric_group_results
        ),
        failure_finding_groups=direct_failure_finding_groups,
    )
