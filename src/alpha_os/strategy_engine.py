from __future__ import annotations

from dataclasses import dataclass

from .evaluation_cost_config import (
    EvaluationRebalanceFrictionPolicySpec,
    ExecutionCostAssumptionsSpec,
    HoldingCostAssumptionsSpec,
)
from .evaluation_spec import EvaluationDateRange
from .portfolio_construction_config import PortfolioConstructionSpec
from .strategy_execution import StrategyExecutionKind
from .strategy_run_mode import StrategyRunMode


@dataclass(frozen=True)
class StrategyEvaluationContext:
    strategy_id: str
    execution_kind: StrategyExecutionKind
    run_mode: StrategyRunMode
    subject_set_id: str
    target_id: str
    base_url: str
    portfolio_construction: PortfolioConstructionSpec
    rebalance_friction_policy: EvaluationRebalanceFrictionPolicySpec
    execution_cost_assumptions: ExecutionCostAssumptionsSpec
    holding_cost_assumptions: HoldingCostAssumptionsSpec


@dataclass(frozen=True)
class StrategyEvaluationArtifacts:
    signal_train_id: str
    initial_strategy_state_id: str | None
    signal_discovery_run_id: str | None
    signal_discovery_id: str | None
    screening_result_id: str | None
    compressed_belief_id: str | None


@dataclass(frozen=True)
class BacktestOosRunInputs:
    evaluation_spec_id: str
    execution_range: EvaluationDateRange
    evaluation_date_ranges: tuple[EvaluationDateRange, ...]
    metric_group_names: tuple[str, ...]


@dataclass(frozen=True)
class FixedStateReplayRunInputs:
    evaluation_spec_id: str
    fixed_initial_strategy_state_id: str
    execution_range: EvaluationDateRange
    evaluation_date_ranges: tuple[EvaluationDateRange, ...]
    metric_group_names: tuple[str, ...]


@dataclass(frozen=True)
class PaperRunInputs:
    as_of_timestamp: str
    current_portfolio_state_id: str | None = None


@dataclass(frozen=True)
class LiveRunInputs:
    as_of_timestamp: str
    venue_id: str
    current_portfolio_state_id: str | None = None


@dataclass(frozen=True, init=False)
class StrategyEvaluationRequest:
    evaluation_task_id: str
    evaluation_spec_id: str
    fold_label: str
    context: StrategyEvaluationContext
    artifacts: StrategyEvaluationArtifacts
    execution_range: EvaluationDateRange
    evaluation_date_ranges: tuple[EvaluationDateRange, ...]
    metric_group_names: tuple[str, ...]

    def __init__(
        self,
        *,
        evaluation_task_id: str | None = None,
        evaluation_spec_id: str,
        fold_label: str,
        context: StrategyEvaluationContext,
        artifacts: StrategyEvaluationArtifacts,
        execution_range: EvaluationDateRange,
        evaluation_date_ranges: tuple[EvaluationDateRange, ...],
        metric_group_names: tuple[str, ...],
    ) -> None:
        if evaluation_task_id is None:
            raise ValueError("strategy evaluation request requires evaluation_task_id")
        object.__setattr__(self, "evaluation_task_id", evaluation_task_id)
        object.__setattr__(self, "evaluation_spec_id", evaluation_spec_id)
        object.__setattr__(self, "fold_label", fold_label)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "execution_range", execution_range)
        object.__setattr__(self, "evaluation_date_ranges", evaluation_date_ranges)
        object.__setattr__(self, "metric_group_names", metric_group_names)

    def to_backtest_oos_run_inputs(self) -> BacktestOosRunInputs:
        if self.context.run_mode != "backtest_oos":
            raise ValueError(
                "backtest_oos run inputs require run_mode=backtest_oos"
            )
        return BacktestOosRunInputs(
            evaluation_spec_id=self.evaluation_spec_id,
            execution_range=self.execution_range,
            evaluation_date_ranges=self.evaluation_date_ranges,
            metric_group_names=self.metric_group_names,
        )

    def to_fixed_state_replay_run_inputs(self) -> FixedStateReplayRunInputs:
        if self.context.run_mode != "fixed_state_replay":
            raise ValueError(
                "fixed_state_replay run inputs require "
                "run_mode=fixed_state_replay"
            )
        fixed_initial_strategy_state_id = self.artifacts.initial_strategy_state_id
        if fixed_initial_strategy_state_id is None:
            raise ValueError(
                "fixed_state_replay run inputs require "
                "initial_strategy_state_id"
            )
        return FixedStateReplayRunInputs(
            evaluation_spec_id=self.evaluation_spec_id,
            fixed_initial_strategy_state_id=fixed_initial_strategy_state_id,
            execution_range=self.execution_range,
            evaluation_date_ranges=self.evaluation_date_ranges,
            metric_group_names=self.metric_group_names,
        )
