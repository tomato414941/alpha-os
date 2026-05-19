from __future__ import annotations

from .evaluation_task import EvaluationTask


def select_evaluation_tasks(
    read_port,
    *,
    evaluation_spec_id: str,
    strategy_ids: tuple[str, ...] | None,
    evaluation_task_ids: tuple[str, ...] | None = None,
) -> tuple[EvaluationTask, ...]:
    existing_tasks = tuple(
        state.task
        for state in read_port.list_evaluation_tasks(
            evaluation_spec_id=evaluation_spec_id,
            limit=10_000,
        )
    )
    if not existing_tasks:
        raise ValueError(
            "evaluation spec requires at least one evaluation task: "
            f"{evaluation_spec_id}"
        )
    if evaluation_task_ids is not None:
        allowed_task_ids = set(evaluation_task_ids)
        existing_tasks = tuple(
            item for item in existing_tasks if item.evaluation_task_id in allowed_task_ids
        )
        if not existing_tasks:
            raise ValueError(
                "evaluation spec does not contain requested evaluation tasks: "
                f"{evaluation_spec_id}"
            )
    if strategy_ids:
        allowed_strategy_ids = set(strategy_ids)
        existing_tasks = tuple(
            item for item in existing_tasks if item.strategy_id in allowed_strategy_ids
        )
        if not existing_tasks:
            raise ValueError(
                "evaluation spec does not contain requested strategies: "
                f"{evaluation_spec_id}"
    )
    return tuple(
        sorted(
            existing_tasks,
            key=lambda item: (
                item.strategy_id,
                item.evaluation_task_id,
            ),
        )
    )
