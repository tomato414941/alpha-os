from __future__ import annotations

from dataclasses import dataclass

from .evaluation_cost_config import (
    EvaluationRebalanceFrictionPolicySpec,
    ExecutionCostAssumptionsSpec,
    HoldingCostAssumptionsSpec,
)
from .evaluation_spec import EvaluationDateRange
from .portfolio_construction_config import PortfolioConstructionSpec


@dataclass(frozen=True)
class StrategyEvaluationContext:
    strategy_id: str
    subject_set_id: str
    target_id: str
    base_url: str
    portfolio_construction: PortfolioConstructionSpec
    rebalance_friction_policy: EvaluationRebalanceFrictionPolicySpec
    execution_cost_assumptions: ExecutionCostAssumptionsSpec
    holding_cost_assumptions: HoldingCostAssumptionsSpec


@dataclass(frozen=True)
class StrategyEvaluationInputRefs:
    strategy_checkpoint_id: str


@dataclass(frozen=True, init=False)
class StrategyEvaluationRequest:
    evaluation_task_id: str
    evaluation_spec_id: str
    fold_label: str
    context: StrategyEvaluationContext
    input_refs: StrategyEvaluationInputRefs | None
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
        input_refs: StrategyEvaluationInputRefs | None,
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
        object.__setattr__(self, "input_refs", input_refs)
        object.__setattr__(self, "execution_range", execution_range)
        object.__setattr__(self, "evaluation_date_ranges", evaluation_date_ranges)
        object.__setattr__(self, "metric_group_names", metric_group_names)
