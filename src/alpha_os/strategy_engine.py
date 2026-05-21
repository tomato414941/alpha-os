from __future__ import annotations

from .evaluation_spec import EvaluationDateRange


class StrategyEvaluationRequest:
    evaluation_task_id: str
    evaluation_spec_id: str
    fold_label: str
    strategy_id: str
    base_url: str
    execution_range: EvaluationDateRange
    evaluation_date_ranges: tuple[EvaluationDateRange, ...]
    metric_group_names: tuple[str, ...]

    def __init__(
        self,
        *,
        evaluation_task_id: str | None = None,
        evaluation_spec_id: str,
        fold_label: str,
        strategy_id: str,
        base_url: str,
        execution_range: EvaluationDateRange,
        evaluation_date_ranges: tuple[EvaluationDateRange, ...],
        metric_group_names: tuple[str, ...],
    ) -> None:
        if evaluation_task_id is None:
            raise ValueError("strategy evaluation request requires evaluation_task_id")
        object.__setattr__(self, "evaluation_task_id", evaluation_task_id)
        object.__setattr__(self, "evaluation_spec_id", evaluation_spec_id)
        object.__setattr__(self, "fold_label", fold_label)
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "execution_range", execution_range)
        object.__setattr__(self, "evaluation_date_ranges", evaluation_date_ranges)
        object.__setattr__(self, "metric_group_names", metric_group_names)
