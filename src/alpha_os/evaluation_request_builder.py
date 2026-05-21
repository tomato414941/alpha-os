from __future__ import annotations

from .evaluation_task import EvaluationTask
from .evaluation_spec import (
    EvaluationSpec,
    EvaluationDateRange,
)
from .evaluation_request import (
    StrategyEvaluationRequest,
)


def _strategy_evaluation_request(
    *,
    evaluation_task_id: str,
    evaluation_spec_id: str,
    fold_label: str,
    strategy_id: str,
    execution_range: EvaluationDateRange,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
    metric_group_names: tuple[str, ...],
    base_url: str,
) -> StrategyEvaluationRequest:
    return StrategyEvaluationRequest(
        evaluation_task_id=evaluation_task_id,
        evaluation_spec_id=evaluation_spec_id,
        fold_label=fold_label,
        strategy_id=strategy_id,
        base_url=base_url,
        execution_range=execution_range,
        evaluation_date_ranges=evaluation_date_ranges,
        metric_group_names=metric_group_names,
    )


def build_strategy_evaluation_requests(
    *,
    evaluation_spec_id: str,
    evaluation_spec: EvaluationSpec,
    evaluation_tasks: tuple[EvaluationTask, ...] | None = None,
    base_url: str,
) -> tuple[StrategyEvaluationRequest, ...]:
    execution_requests: list[StrategyEvaluationRequest] = []
    if evaluation_tasks is None:
        raise ValueError("strategy evaluation request builder requires evaluation_tasks")
    for evaluation_task in evaluation_tasks:
        for fold in evaluation_spec.resolved_evaluation_folds:
            execution_requests.append(
                _strategy_evaluation_request(
                    evaluation_task_id=evaluation_task.evaluation_task_id,
                    evaluation_spec_id=evaluation_spec_id,
                    fold_label=fold.label,
                    strategy_id=evaluation_task.strategy_id,
                    execution_range=fold.execution_range,
                    evaluation_date_ranges=fold.resolved_evaluation_date_ranges,
                    metric_group_names=evaluation_spec.metric_group_names,
                    base_url=base_url,
                )
            )
    return tuple(execution_requests)
