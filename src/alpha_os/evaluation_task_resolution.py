from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .evaluation_task import EvaluationTask
from .store import EvaluationStore


class EvaluationTaskResolutionReadPort(Protocol):
    def list_evaluation_tasks(
        self,
        *,
        evaluation_spec_id: str | None = None,
        limit: int = 100,
    ):
        ...

    def get_trading_strategy(self, strategy_id: str):
        ...


@dataclass(frozen=True)
class EvaluationTaskResolutionRequest:
    evaluation_spec_id: str
    strategy_ids: tuple[str, ...] | None
    evaluation_task_ids: tuple[str, ...] | None = None


def resolve_evaluation_tasks_for_request(
    read_port: EvaluationTaskResolutionReadPort,
    request: EvaluationTaskResolutionRequest,
) -> tuple[EvaluationTask, ...]:
    evaluation_spec_id = request.evaluation_spec_id
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
    if request.evaluation_task_ids is not None:
        allowed_task_ids = set(request.evaluation_task_ids)
        existing_tasks = tuple(
            item for item in existing_tasks if item.evaluation_task_id in allowed_task_ids
        )
        if not existing_tasks:
            raise ValueError(
                "evaluation spec does not contain requested evaluation tasks: "
                f"{evaluation_spec_id}"
            )
    if request.strategy_ids:
        allowed_strategy_ids = set(request.strategy_ids)
        existing_tasks = tuple(
            item for item in existing_tasks if item.strategy_id in allowed_strategy_ids
        )
        if not existing_tasks:
            raise ValueError(
                "evaluation spec does not contain requested strategies: "
                f"{evaluation_spec_id}"
            )
    for task in existing_tasks:
        source_strategy_state = read_port.get_trading_strategy(task.strategy_id)
        if source_strategy_state is None:
            raise ValueError(
                f"evaluation task strategy does not exist: {task.strategy_id}"
            )
    unique_tasks: dict[str, EvaluationTask] = {}
    for task in existing_tasks:
        unique_tasks.setdefault(task.evaluation_task_id, task)
    return tuple(
        sorted(
            unique_tasks.values(),
            key=lambda item: (
                item.strategy_id,
                item.evaluation_task_id,
            ),
        )
    )


def resolve_evaluation_tasks_for_spec(
    store: EvaluationStore,
    *,
    evaluation_spec_state,
    strategy_ids: tuple[str, ...] | None,
    evaluation_task_ids: tuple[str, ...] | None = None,
) -> tuple[EvaluationTask, ...]:
    return resolve_evaluation_tasks_for_request(
        store,
        EvaluationTaskResolutionRequest(
            evaluation_spec_id=evaluation_spec_state.evaluation_spec_id,
            strategy_ids=strategy_ids,
            evaluation_task_ids=evaluation_task_ids,
        ),
    )
