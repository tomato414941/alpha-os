from __future__ import annotations

from .data_repositories import (
    FeaturePlaneRepository,
    ObservationFrameRepository,
)
from .evaluation_task_query import select_evaluation_tasks
from .evaluation_runner import EvaluationRunRequest, evaluate_evaluation_spec_state
from .store import EvaluationStore


def run_evaluation_use_case(
    *,
    store: EvaluationStore,
    evaluation_spec_id: str,
    strategy_ids: tuple[str, ...] | None,
    base_url: str,
):
    store.ensure_schema()
    feature_plane_repository = FeaturePlaneRepository(
        observation_repository=ObservationFrameRepository(store=store)
    )
    evaluation_spec_state = store.get_evaluation_spec(evaluation_spec_id)
    if evaluation_spec_state is None:
        raise ValueError(f"evaluation spec does not exist: {evaluation_spec_id}")
    evaluation_tasks = select_evaluation_tasks(
        store,
        evaluation_spec_id=evaluation_spec_state.evaluation_spec_id,
        strategy_ids=strategy_ids,
    )
    return evaluate_evaluation_spec_state(
        EvaluationRunRequest(
            store=store,
            evaluation_spec_state=evaluation_spec_state,
            evaluation_tasks=evaluation_tasks,
            base_url=base_url,
            feature_plane_repository=feature_plane_repository,
        )
    )


def run_walk_forward_evaluation_use_case(
    *,
    store: EvaluationStore,
    evaluation_spec_id: str,
    strategy_ids: tuple[str, ...] | None,
    evaluation_task_ids: tuple[str, ...] | None,
    base_url: str,
):
    store.ensure_schema()
    feature_plane_repository = FeaturePlaneRepository(
        observation_repository=ObservationFrameRepository(store=store)
    )
    evaluation_spec_state = store.get_evaluation_spec(evaluation_spec_id)
    if evaluation_spec_state is None:
        raise ValueError(f"evaluation spec does not exist: {evaluation_spec_id}")
    evaluation_tasks = select_evaluation_tasks(
        store,
        evaluation_spec_id=evaluation_spec_state.evaluation_spec_id,
        strategy_ids=strategy_ids,
        evaluation_task_ids=evaluation_task_ids,
    )
    return evaluate_evaluation_spec_state(
        EvaluationRunRequest(
            store=store,
            evaluation_spec_state=evaluation_spec_state,
            evaluation_tasks=evaluation_tasks,
            base_url=base_url,
            feature_plane_repository=feature_plane_repository,
        )
    )
