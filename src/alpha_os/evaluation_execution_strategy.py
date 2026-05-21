from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .data_repositories import FeaturePlaneRepository
from .evaluation_cost_config import (
    EvaluationRebalanceFrictionPolicySpec,
    ExecutionCostAssumptionsSpec,
    HoldingCostAssumptionsSpec,
)
from .strategy_backtest import run_strategy_backtest_from_store
from .strategy_engine import StrategyEvaluationRequest
from .portfolio_construction_config import PortfolioConstructionSpec
from .evaluation_result import EvaluationTaskResult
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
    strategy_state = store.get_trading_strategy(execution_request.strategy_id)
    if strategy_state is None:
        raise ValueError(
            "evaluation task strategy does not exist: "
            f"{execution_request.strategy_id}"
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


def _target_id_for_strategy(trading_strategy: TradingStrategySpec) -> str:
    target_id = trading_strategy.target_id
    if not isinstance(target_id, str) or not target_id:
        raise ValueError(
            "evaluation task strategy is missing prediction target: "
            f"{trading_strategy.strategy_id}"
        )
    return target_id


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
    target_id = _target_id_for_strategy(trading_strategy)
    subject_set_state = store.get_subject_set(subject_set_id)
    if subject_set_state is not None:
        validate_subject_set_universe_contract(subject_set_state.definition)
    direct_evaluation = run_strategy_backtest_from_store(
        store=store,
        strategy_id=execution_request.strategy_id,
        subject_set_id=subject_set_id,
        target_id=target_id,
        evaluation_date_ranges=execution_request.evaluation_date_ranges,
        base_url=execution_request.base_url,
        portfolio_construction=portfolio_construction,
        rebalance_friction_policy=rebalance_friction_policy,
        execution_cost_assumptions=execution_cost_assumptions,
        holding_cost_assumptions=holding_cost_assumptions,
        feature_plane_repository=context.feature_plane_repository,
    )
    direct_metric_group_results, direct_failure_finding_groups = direct_evaluation
    return EvaluationTaskResult(
        evaluation_task_id=execution_request.evaluation_task_id,
        strategy_id=execution_request.strategy_id,
        metric_group_results=tuple(
            direct_metric_group_results[metric_group_name]
            for metric_group_name in execution_request.metric_group_names
            if metric_group_name in direct_metric_group_results
        ),
        failure_finding_groups=direct_failure_finding_groups,
    )
