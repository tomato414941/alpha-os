from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .evaluation_task import EvaluationTask
from .evaluation_cost_config import (
    EvaluationRebalanceFrictionPolicySpec,
    ExecutionCostAssumptionsSpec,
    HoldingCostAssumptionsSpec,
)
from .evaluation_spec import (
    EvaluationSpec,
    EvaluationDateRange,
)
from .portfolio_construction_config import PortfolioConstructionSpec
from .strategy_engine import (
    StrategyEvaluationInputRefs,
    StrategyEvaluationContext,
    StrategyEvaluationRequest,
)

if TYPE_CHECKING:
    from .store import (
        StrategyCheckpointRecord,
        TradingStrategyState,
    )


class EvaluationPlanReadPort(Protocol):
    def get_trading_strategy(
        self,
        strategy_id: str,
    ) -> TradingStrategyState | None: ...

    def get_strategy_checkpoint(
        self,
        strategy_checkpoint_id: str,
    ) -> StrategyCheckpointRecord | None: ...

    def list_strategy_checkpoints(
        self,
        *,
        strategy_id: str | None = None,
        signal_discovery_id: str | None = None,
        fold_label: str | None = None,
        execution_start_date: str | None = None,
        execution_end_date: str | None = None,
        limit: int = 20,
    ) -> list[StrategyCheckpointRecord]: ...


@dataclass(frozen=True)
class EvaluationPlan:
    evaluation_spec_id: str
    metric_group_names: tuple[str, ...]
    execution_requests: tuple[StrategyEvaluationRequest, ...]


def _strategy_evaluation_request(
    *,
    evaluation_task_id: str,
    evaluation_spec_id: str,
    fold_label: str,
    strategy_id: str,
    signal_discovery_id: str | None,
    strategy_checkpoint_id: str | None,
    snapshot_set_id: str | None,
    prepared_start_date: str | None,
    prepared_end_date: str | None,
    subject_set_id: str,
    target_id: str,
    screening_result_id: str | None,
    compressed_belief_id: str | None,
    execution_range: EvaluationDateRange,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
    metric_group_names: tuple[str, ...],
    base_url: str,
    selection_kind: str,
    top_k: int | None,
    portfolio_construction: PortfolioConstructionSpec,
    rebalance_friction_policy: EvaluationRebalanceFrictionPolicySpec,
    execution_cost_assumptions: ExecutionCostAssumptionsSpec,
    holding_cost_assumptions: HoldingCostAssumptionsSpec,
) -> StrategyEvaluationRequest:
    input_refs = None
    if (
        strategy_checkpoint_id is not None
        or snapshot_set_id is not None
        or screening_result_id is not None
        or compressed_belief_id is not None
    ):
        if prepared_start_date is None or prepared_end_date is None:
            raise ValueError("prepared evaluation inputs require prepared date range")
        input_refs = StrategyEvaluationInputRefs(
            strategy_checkpoint_id=strategy_checkpoint_id,
            snapshot_set_id=snapshot_set_id,
            screening_result_id=screening_result_id,
            compressed_belief_id=compressed_belief_id,
            prepared_start_date=prepared_start_date,
            prepared_end_date=prepared_end_date,
        )
    return StrategyEvaluationRequest(
        evaluation_task_id=evaluation_task_id,
        evaluation_spec_id=evaluation_spec_id,
        fold_label=fold_label,
        context=StrategyEvaluationContext(
            strategy_id=strategy_id,
            signal_discovery_id=signal_discovery_id,
            subject_set_id=subject_set_id,
            target_id=target_id,
            base_url=base_url,
            selection_kind=selection_kind,
            top_k=top_k,
            portfolio_construction=portfolio_construction,
            rebalance_friction_policy=rebalance_friction_policy,
            execution_cost_assumptions=execution_cost_assumptions,
            holding_cost_assumptions=holding_cost_assumptions,
        ),
        input_refs=input_refs,
        execution_range=execution_range,
        evaluation_date_ranges=evaluation_date_ranges,
        metric_group_names=metric_group_names,
    )


def _rebalance_interval_steps_from_strategy(
    rebalance: str | None,
    rebalance_interval_steps: int | None,
) -> int | None:
    if rebalance_interval_steps is not None:
        return int(rebalance_interval_steps)
    if rebalance in {None, "", "-", "none"}:
        return None
    prefix = "every_"
    suffix = "_steps"
    if not rebalance.startswith(prefix) or not rebalance.endswith(suffix):
        raise ValueError(f"unsupported strategy rebalance policy: {rebalance}")
    value = rebalance[len(prefix) : -len(suffix)]
    try:
        steps = int(value)
    except ValueError as exc:
        raise ValueError(f"unsupported strategy rebalance policy: {rebalance}") from exc
    if steps < 1:
        raise ValueError(f"unsupported strategy rebalance policy: {rebalance}")
    return steps


def _portfolio_construction_for_strategy(
    *,
    trading_strategy,
) -> PortfolioConstructionSpec:
    return trading_strategy.portfolio.portfolio_construction


def _rebalance_friction_policy_for_strategy(
    *,
    trading_strategy,
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
    *,
    trading_strategy,
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
    *,
    trading_strategy,
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


def build_evaluation_plan(
    store: EvaluationPlanReadPort,
    *,
    evaluation_spec_id: str,
    evaluation_spec: EvaluationSpec,
    evaluation_tasks: tuple[EvaluationTask, ...] | None = None,
    default_target_id: str,
    base_url: str,
) -> EvaluationPlan:
    execution_requests: list[StrategyEvaluationRequest] = []
    if evaluation_tasks is None:
        raise ValueError("evaluation plan requires evaluation_tasks")
    for evaluation_task in evaluation_tasks:
        strategy_state = store.get_trading_strategy(evaluation_task.strategy_id)
        if strategy_state is None:
            raise ValueError(
                "evaluation task strategy does not exist: "
                f"{evaluation_task.strategy_id}"
            )
        trading_strategy = strategy_state.trading_strategy
        portfolio_construction = _portfolio_construction_for_strategy(
            trading_strategy=trading_strategy,
        )
        rebalance_friction_policy = _rebalance_friction_policy_for_strategy(
            trading_strategy=trading_strategy,
        )
        execution_cost_assumptions = _execution_cost_assumptions_for_strategy(
            trading_strategy=trading_strategy,
        )
        holding_cost_assumptions = _holding_cost_assumptions_for_strategy(
            trading_strategy=trading_strategy,
        )
        strategy_signal_discovery_id = trading_strategy.signal_discovery_id
        if strategy_signal_discovery_id is None:
            subject_set_id = trading_strategy.subject_set_id
            if not isinstance(subject_set_id, str) or not subject_set_id:
                raise ValueError(
                    "direct evaluation task requires strategy subject_set: "
                    f"{evaluation_task.evaluation_task_id}"
                )
            target_id = trading_strategy.target_id or default_target_id
            for fold in evaluation_spec.resolved_evaluation_folds:
                execution_requests.append(
                    _strategy_evaluation_request(
                        evaluation_task_id=evaluation_task.evaluation_task_id,
                        evaluation_spec_id=evaluation_spec_id,
                        fold_label=fold.label,
                        strategy_id=evaluation_task.strategy_id,
                        signal_discovery_id=None,
                        strategy_checkpoint_id=None,
                        snapshot_set_id=None,
                        prepared_start_date=None,
                        prepared_end_date=None,
                        subject_set_id=subject_set_id,
                        target_id=target_id,
                        screening_result_id=None,
                        compressed_belief_id=None,
                        execution_range=fold.execution_range,
                        evaluation_date_ranges=fold.resolved_evaluation_date_ranges,
                        metric_group_names=evaluation_spec.metric_group_names,
                        base_url=base_url,
                        selection_kind=trading_strategy.selection_kind,
                        top_k=trading_strategy.portfolio.top_k,
                        portfolio_construction=portfolio_construction,
                        rebalance_friction_policy=rebalance_friction_policy,
                        execution_cost_assumptions=execution_cost_assumptions,
                        holding_cost_assumptions=holding_cost_assumptions,
                    )
                )
            continue
        for fold in evaluation_spec.resolved_evaluation_folds:
            strategy_checkpoints = store.list_strategy_checkpoints(
                strategy_id=evaluation_task.strategy_id,
                signal_discovery_id=strategy_signal_discovery_id,
                fold_label=fold.label,
                execution_start_date=fold.execution_range.start_date,
                execution_end_date=fold.execution_range.end_date,
                limit=1,
            )
            if not strategy_checkpoints:
                strategy_checkpoints = store.list_strategy_checkpoints(
                    strategy_id=evaluation_task.strategy_id,
                    signal_discovery_id=strategy_signal_discovery_id,
                    execution_start_date=fold.execution_range.start_date,
                    execution_end_date=fold.execution_range.end_date,
                    limit=1,
                )
            if strategy_checkpoints:
                strategy_checkpoint = strategy_checkpoints[0].state
                strategy_checkpoint_id = (
                    strategy_checkpoint.strategy_checkpoint_id
                )
                snapshot_set_id = strategy_checkpoint.snapshot_set_id
                prepared_start_date = strategy_checkpoint.execution_start_date
                prepared_end_date = strategy_checkpoint.execution_end_date
                subject_set_id = strategy_checkpoint.subject_set_id
                target_id = strategy_checkpoint.target_id
                screening_result_id = strategy_checkpoint.screening_result_id
                compressed_belief_id = strategy_checkpoint.compressed_belief_id
            else:
                raise ValueError(
                    "checkpoint evaluation task requires a strategy checkpoint for "
                    f"{evaluation_task.evaluation_task_id} "
                    f"{fold.execution_range.start_date}->{fold.execution_range.end_date}"
                )
            execution_requests.append(
                _strategy_evaluation_request(
                    evaluation_task_id=evaluation_task.evaluation_task_id,
                    evaluation_spec_id=evaluation_spec_id,
                    fold_label=fold.label,
                    strategy_id=evaluation_task.strategy_id,
                    signal_discovery_id=strategy_signal_discovery_id,
                    strategy_checkpoint_id=strategy_checkpoint_id,
                    snapshot_set_id=snapshot_set_id,
                    prepared_start_date=prepared_start_date,
                    prepared_end_date=prepared_end_date,
                    subject_set_id=subject_set_id,
                    target_id=target_id,
                    screening_result_id=screening_result_id,
                    compressed_belief_id=compressed_belief_id,
                    execution_range=fold.execution_range,
                    evaluation_date_ranges=fold.resolved_evaluation_date_ranges,
                    metric_group_names=evaluation_spec.metric_group_names,
                    base_url=base_url,
                    selection_kind=trading_strategy.selection_kind,
                    top_k=trading_strategy.portfolio.top_k,
                    portfolio_construction=portfolio_construction,
                    rebalance_friction_policy=rebalance_friction_policy,
                    execution_cost_assumptions=execution_cost_assumptions,
                    holding_cost_assumptions=holding_cost_assumptions,
                )
            )
    return EvaluationPlan(
        evaluation_spec_id=evaluation_spec_id,
        metric_group_names=evaluation_spec.metric_group_names,
        execution_requests=tuple(execution_requests),
    )
