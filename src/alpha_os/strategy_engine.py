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
    signal_discovery_id: str | None
    subject_set_id: str
    target_id: str
    base_url: str
    selection_kind: str
    top_k: int | None
    portfolio_construction: PortfolioConstructionSpec
    rebalance_friction_policy: EvaluationRebalanceFrictionPolicySpec
    execution_cost_assumptions: ExecutionCostAssumptionsSpec
    holding_cost_assumptions: HoldingCostAssumptionsSpec


@dataclass(frozen=True)
class StrategyEvaluationInputRefs:
    strategy_checkpoint_id: str | None
    snapshot_set_id: str | None
    screening_result_id: str | None
    compressed_belief_id: str | None
    prepared_start_date: str
    prepared_end_date: str


@dataclass(frozen=True)
class StrategyEvaluationDiagnosticRefs:
    signal_discovery_run_id: str | None


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
    input_refs: StrategyEvaluationInputRefs | None
    diagnostic_refs: StrategyEvaluationDiagnosticRefs | None
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
        diagnostic_refs: StrategyEvaluationDiagnosticRefs | None = None,
    ) -> None:
        if evaluation_task_id is None:
            raise ValueError("strategy evaluation request requires evaluation_task_id")
        object.__setattr__(self, "evaluation_task_id", evaluation_task_id)
        object.__setattr__(self, "evaluation_spec_id", evaluation_spec_id)
        object.__setattr__(self, "fold_label", fold_label)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "input_refs", input_refs)
        object.__setattr__(self, "diagnostic_refs", diagnostic_refs)
        object.__setattr__(self, "execution_range", execution_range)
        object.__setattr__(self, "evaluation_date_ranges", evaluation_date_ranges)
        object.__setattr__(self, "metric_group_names", metric_group_names)
