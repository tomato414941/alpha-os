from __future__ import annotations

from dataclasses import dataclass

from .data_repositories import (
    FeaturePlaneRepository,
    ObservationFrameRepository,
)
from .evaluation_task_query import select_evaluation_tasks
from .evaluation_runner import EvaluationRunRequest, evaluate_evaluation_spec_state
from .store import EvaluationStore


@dataclass(frozen=True)
class RunEvaluationUseCaseRequest:
    store: EvaluationStore
    evaluation_spec_id: str
    strategy_ids: tuple[str, ...] | None
    base_url: str


@dataclass(frozen=True)
class RunEvaluationUseCaseResult:
    report_state: object


@dataclass(frozen=True)
class RunWalkForwardEvaluationUseCaseRequest:
    store: EvaluationStore
    evaluation_spec_id: str
    strategy_ids: tuple[str, ...] | None
    evaluation_task_ids: tuple[str, ...] | None
    base_url: str


@dataclass(frozen=True)
class RunWalkForwardEvaluationUseCaseResult:
    report_state: object


def run_evaluation_use_case(
    request: RunEvaluationUseCaseRequest,
) -> RunEvaluationUseCaseResult:
    store = request.store
    store.ensure_schema()
    feature_plane_repository = FeaturePlaneRepository(
        observation_repository=ObservationFrameRepository(store=store)
    )
    evaluation_spec_state = store.get_evaluation_spec(request.evaluation_spec_id)
    if evaluation_spec_state is None:
        raise ValueError(f"evaluation spec does not exist: {request.evaluation_spec_id}")
    evaluation_tasks = select_evaluation_tasks(
        store,
        evaluation_spec_id=evaluation_spec_state.evaluation_spec_id,
        strategy_ids=request.strategy_ids,
    )
    report_state = evaluate_evaluation_spec_state(
        EvaluationRunRequest(
            store=store,
            evaluation_spec_state=evaluation_spec_state,
            evaluation_tasks=evaluation_tasks,
            base_url=request.base_url,
            feature_plane_repository=feature_plane_repository,
        )
    )
    return RunEvaluationUseCaseResult(report_state=report_state)


def run_walk_forward_evaluation_use_case(
    request: RunWalkForwardEvaluationUseCaseRequest,
) -> RunWalkForwardEvaluationUseCaseResult:
    store = request.store
    store.ensure_schema()
    feature_plane_repository = FeaturePlaneRepository(
        observation_repository=ObservationFrameRepository(store=store)
    )
    evaluation_spec_state = store.get_evaluation_spec(request.evaluation_spec_id)
    if evaluation_spec_state is None:
        raise ValueError(f"evaluation spec does not exist: {request.evaluation_spec_id}")
    evaluation_tasks = select_evaluation_tasks(
        store,
        evaluation_spec_id=evaluation_spec_state.evaluation_spec_id,
        strategy_ids=request.strategy_ids,
        evaluation_task_ids=request.evaluation_task_ids,
    )
    report_state = evaluate_evaluation_spec_state(
        EvaluationRunRequest(
            store=store,
            evaluation_spec_state=evaluation_spec_state,
            evaluation_tasks=evaluation_tasks,
            base_url=request.base_url,
            feature_plane_repository=feature_plane_repository,
        )
    )
    return RunWalkForwardEvaluationUseCaseResult(report_state=report_state)
