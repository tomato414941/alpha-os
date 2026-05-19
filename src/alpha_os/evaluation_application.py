from __future__ import annotations

from dataclasses import dataclass

from .data_repositories import (
    FeaturePlaneRepository,
    ObservationFrameRepository,
)
from .evaluation_task_resolution import resolve_evaluation_tasks_for_spec
from .evaluation_runner import EvaluationRunRequest, evaluate_evaluation_spec_state
from .store import EvaluationStore


@dataclass(frozen=True)
class RunEvaluationUseCaseRequest:
    store: EvaluationStore
    evaluation_spec_id: str
    sizing_method: str | None
    sizing_engine: str | None
    direction_mode: str | None
    strategy_ids: tuple[str, ...] | None
    base_url: str
    created_at: str


@dataclass(frozen=True)
class RunEvaluationUseCaseResult:
    report_state: object


@dataclass(frozen=True)
class RunWalkForwardEvaluationUseCaseRequest:
    store: EvaluationStore
    evaluation_spec_id: str
    sizing_method: str | None
    sizing_engine: str | None
    direction_mode: str | None
    strategy_ids: tuple[str, ...] | None
    evaluation_task_ids: tuple[str, ...] | None
    base_url: str
    created_at: str


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
    evaluation_tasks = resolve_evaluation_tasks_for_spec(
        store,
        evaluation_spec_state=evaluation_spec_state,
        sizing_method=request.sizing_method,
        sizing_engine=request.sizing_engine,
        direction_mode=request.direction_mode,
        strategy_ids=request.strategy_ids,
        created_at=request.created_at,
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
    evaluation_tasks = resolve_evaluation_tasks_for_spec(
        store,
        evaluation_spec_state=evaluation_spec_state,
        sizing_method=request.sizing_method,
        sizing_engine=request.sizing_engine,
        direction_mode=request.direction_mode,
        strategy_ids=request.strategy_ids,
        evaluation_task_ids=request.evaluation_task_ids,
        created_at=request.created_at,
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
