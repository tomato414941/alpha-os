from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .evaluation_task import EvaluationTask
from .strategy_training import build_signal_train_id
from .evaluation_cost_config import (
    EvaluationRebalanceFrictionPolicySpec,
    ExecutionCostAssumptionsSpec,
    HoldingCostAssumptionsSpec,
)
from .evaluation_job_spec import EvaluationJobSpec, default_evaluation_job_spec
from .evaluation_spec import (
    EvaluationSpec,
    EvaluationDateRange,
)
from .portfolio_construction_config import PortfolioConstructionSpec
from .strategy_engine import (
    StrategyEvaluationArtifacts,
    StrategyEvaluationContext,
    StrategyEvaluationRequest,
)
from .strategy_run_mode import StrategyRunMode

if TYPE_CHECKING:
    from .store import (
        InitialStrategyStateRecord,
        SignalDiscoveryRunState,
        TradingStrategyState,
    )


class EvaluationPlanReadPort(Protocol):
    def get_trading_strategy(
        self,
        strategy_id: str,
    ) -> TradingStrategyState | None: ...

    def get_initial_strategy_state(
        self,
        initial_strategy_state_id: str,
    ) -> InitialStrategyStateRecord | None: ...

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
    ) -> list[InitialStrategyStateRecord]: ...

    def list_signal_discovery_runs(
        self,
        *,
        signal_discovery_id: str | None = None,
        execution_start_date: str | None = None,
        execution_end_date: str | None = None,
        limit: int = 20,
    ) -> list[SignalDiscoveryRunState]: ...


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
    execution_kind: str,
    run_mode: StrategyRunMode,
    signal_train_id: str,
    initial_strategy_state_id: str | None,
    signal_discovery_run_id: str | None,
    signal_discovery_id: str | None,
    subject_set_id: str,
    target_id: str,
    screening_result_id: str | None,
    compressed_belief_id: str | None,
    execution_range: EvaluationDateRange,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
    metric_group_names: tuple[str, ...],
    base_url: str,
    portfolio_construction: PortfolioConstructionSpec,
    rebalance_friction_policy: EvaluationRebalanceFrictionPolicySpec,
    execution_cost_assumptions: ExecutionCostAssumptionsSpec,
    holding_cost_assumptions: HoldingCostAssumptionsSpec,
) -> StrategyEvaluationRequest:
    return StrategyEvaluationRequest(
        evaluation_task_id=evaluation_task_id,
        evaluation_spec_id=evaluation_spec_id,
        fold_label=fold_label,
        context=StrategyEvaluationContext(
            strategy_id=strategy_id,
            execution_kind=execution_kind,
            run_mode=run_mode,
            subject_set_id=subject_set_id,
            target_id=target_id,
            base_url=base_url,
            portfolio_construction=portfolio_construction,
            rebalance_friction_policy=rebalance_friction_policy,
            execution_cost_assumptions=execution_cost_assumptions,
            holding_cost_assumptions=holding_cost_assumptions,
        ),
        artifacts=StrategyEvaluationArtifacts(
            signal_train_id=signal_train_id,
            initial_strategy_state_id=initial_strategy_state_id,
            signal_discovery_run_id=signal_discovery_run_id,
            signal_discovery_id=signal_discovery_id,
            screening_result_id=screening_result_id,
            compressed_belief_id=compressed_belief_id,
        ),
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
    evaluation_job_specs: tuple[EvaluationJobSpec, ...] | None = None,
    default_target_id: str,
    base_url: str,
) -> EvaluationPlan:
    execution_requests: list[StrategyEvaluationRequest] = []
    if evaluation_tasks is None:
        raise ValueError("evaluation plan requires evaluation_tasks")
    job_specs_by_case_id = (
        {
            case.evaluation_task_id: default_evaluation_job_spec(
                case.evaluation_task_id
            )
            for case in evaluation_tasks
        }
        if evaluation_job_specs is None
        else {job.evaluation_task_id: job for job in evaluation_job_specs}
    )
    for evaluation_task in evaluation_tasks:
        job_spec = job_specs_by_case_id.get(evaluation_task.evaluation_task_id)
        if job_spec is None:
            raise ValueError(
                "evaluation job spec does not exist for case: "
                f"{evaluation_task.evaluation_task_id}"
            )
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
        execution = trading_strategy.execution
        run_mode = job_spec.run_mode
        strategy_signal_discovery_id = trading_strategy.signal_discovery_id
        strategy_signal_train_id = build_signal_train_id(
            signal_discovery_id=strategy_signal_discovery_id,
        )
        if execution.kind == "trainless":
            subject_set_id = trading_strategy.subject_set_id
            if not isinstance(subject_set_id, str) or not subject_set_id:
                raise ValueError(
                    "trainless evaluation task requires strategy subject_set or universe axis: "
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
                        execution_kind=execution.kind,
                        run_mode=run_mode,
                        signal_train_id=strategy_signal_train_id,
                        initial_strategy_state_id=None,
                        signal_discovery_run_id=None,
                        signal_discovery_id=None,
                        subject_set_id=subject_set_id,
                        target_id=target_id,
                        screening_result_id=None,
                        compressed_belief_id=None,
                        execution_range=fold.execution_range,
                        evaluation_date_ranges=fold.resolved_evaluation_date_ranges,
                        metric_group_names=evaluation_spec.metric_group_names,
                        base_url=base_url,
                        portfolio_construction=portfolio_construction,
                        rebalance_friction_policy=rebalance_friction_policy,
                        execution_cost_assumptions=execution_cost_assumptions,
                        holding_cost_assumptions=holding_cost_assumptions,
                    )
                )
            continue
        if run_mode == "fixed_state_replay":
            fixed_initial_strategy_state_id = job_spec.fixed_initial_strategy_state_id
            if not fixed_initial_strategy_state_id:
                raise ValueError(
                    "fixed-state replay evaluation task requires fixed_initial_strategy_state_id: "
                    f"{evaluation_task.evaluation_task_id}"
                )
            frozen_state_record = store.get_initial_strategy_state(
                fixed_initial_strategy_state_id
            )
            if frozen_state_record is None:
                raise ValueError(
                    "fixed-state replay evaluation task references unknown initial strategy state: "
                    f"{fixed_initial_strategy_state_id}"
                )
            frozen_state = frozen_state_record.state
            for fold in evaluation_spec.resolved_evaluation_folds:
                execution_requests.append(
                    _strategy_evaluation_request(
                        evaluation_task_id=evaluation_task.evaluation_task_id,
                        evaluation_spec_id=evaluation_spec_id,
                        fold_label=fold.label,
                        strategy_id=evaluation_task.strategy_id,
                        execution_kind=execution.kind,
                        run_mode=run_mode,
                        signal_train_id=frozen_state.signal_train_id,
                        initial_strategy_state_id=(
                            frozen_state.initial_strategy_state_id
                        ),
                        signal_discovery_run_id=frozen_state.signal_discovery_run_id,
                        signal_discovery_id=frozen_state.signal_discovery_id,
                        subject_set_id=frozen_state.subject_set_id,
                        target_id=frozen_state.target_id,
                        screening_result_id=frozen_state.screening_result_id,
                        compressed_belief_id=frozen_state.compressed_belief_id,
                        execution_range=fold.execution_range,
                        evaluation_date_ranges=fold.resolved_evaluation_date_ranges,
                        metric_group_names=evaluation_spec.metric_group_names,
                        base_url=base_url,
                        portfolio_construction=portfolio_construction,
                        rebalance_friction_policy=rebalance_friction_policy,
                        execution_cost_assumptions=execution_cost_assumptions,
                        holding_cost_assumptions=holding_cost_assumptions,
                    )
                )
            continue
        if strategy_signal_discovery_id is None:
            raise ValueError(
                "trained evaluation task requires signal discovery provenance: "
                f"{evaluation_task.evaluation_task_id}"
            )
        for fold in evaluation_spec.resolved_evaluation_folds:
            initial_strategy_states = store.list_initial_strategy_states(
                strategy_id=evaluation_task.strategy_id,
                signal_train_id=strategy_signal_train_id,
                fold_label=fold.label,
                execution_start_date=fold.execution_range.start_date,
                execution_end_date=fold.execution_range.end_date,
                limit=1,
            )
            if initial_strategy_states:
                initial_strategy_state = initial_strategy_states[0].state
                initial_strategy_state_id = (
                    initial_strategy_state.initial_strategy_state_id
                )
                signal_discovery_run_id = initial_strategy_state.signal_discovery_run_id
                subject_set_id = initial_strategy_state.subject_set_id
                target_id = initial_strategy_state.target_id
                screening_result_id = initial_strategy_state.screening_result_id
                compressed_belief_id = initial_strategy_state.compressed_belief_id
            else:
                signal_discovery_runs = store.list_signal_discovery_runs(
                    signal_discovery_id=strategy_signal_discovery_id,
                    execution_start_date=fold.execution_range.start_date,
                    execution_end_date=fold.execution_range.end_date,
                    limit=1,
                )
                if not signal_discovery_runs:
                    raise ValueError(
                        "evaluation task requires an existing signal discovery run or initial strategy state for "
                        f"{evaluation_task.evaluation_task_id} "
                        f"{fold.execution_range.start_date}->{fold.execution_range.end_date}"
                    )
                signal_discovery_run = signal_discovery_runs[0].run
                initial_strategy_state_id = None
                signal_discovery_run_id = signal_discovery_run.signal_discovery_run_id
                subject_set_id = signal_discovery_run.subject_set_id
                target_id = signal_discovery_run.target_id or default_target_id
                screening_result_id = signal_discovery_run.screening_result_id
                compressed_belief_id = signal_discovery_run.compressed_belief_id
            execution_requests.append(
                _strategy_evaluation_request(
                    evaluation_task_id=evaluation_task.evaluation_task_id,
                    evaluation_spec_id=evaluation_spec_id,
                    fold_label=fold.label,
                    strategy_id=evaluation_task.strategy_id,
                    execution_kind=execution.kind,
                    run_mode=run_mode,
                    signal_train_id=strategy_signal_train_id,
                    initial_strategy_state_id=initial_strategy_state_id,
                    signal_discovery_run_id=signal_discovery_run_id,
                    signal_discovery_id=strategy_signal_discovery_id,
                    subject_set_id=subject_set_id,
                    target_id=target_id,
                    screening_result_id=screening_result_id,
                    compressed_belief_id=compressed_belief_id,
                    execution_range=fold.execution_range,
                    evaluation_date_ranges=fold.resolved_evaluation_date_ranges,
                    metric_group_names=evaluation_spec.metric_group_names,
                    base_url=base_url,
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
