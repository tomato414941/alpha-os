from __future__ import annotations

from .data_repositories import (
    FeaturePlaneRepository,
    ObservationFrameRepository,
)
from .evaluation_runner import EvaluationCase, evaluate_evaluation_spec_state
from .store import EvaluationStore


def _build_evaluation_result_key(
    *,
    evaluation_spec_id: str,
    strategy_id: str,
) -> str:
    return f"{evaluation_spec_id}:{strategy_id}"


def _evaluation_cases_for_strategy_ids(
    *,
    evaluation_spec_id: str,
    strategy_ids: tuple[str, ...] | None,
) -> tuple[EvaluationCase, ...]:
    if not strategy_ids:
        raise ValueError(
            "evaluation requires at least one strategy_id when no manifest cases "
            "are provided"
        )
    return tuple(
        (
            _build_evaluation_result_key(
                evaluation_spec_id=evaluation_spec_id,
                strategy_id=strategy_id,
            ),
            strategy_id,
        )
        for strategy_id in strategy_ids
    )


def _all_strategy_ids(store: EvaluationStore) -> tuple[str, ...]:
    return tuple(
        state.strategy_id
        for state in sorted(
            store.list_trading_strategies(limit=10_000),
            key=lambda item: item.strategy_id,
        )
    )


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
    if not strategy_ids:
        strategy_ids = _all_strategy_ids(store)
    evaluation_cases = _evaluation_cases_for_strategy_ids(
        evaluation_spec_id=evaluation_spec_state.evaluation_spec_id,
        strategy_ids=strategy_ids,
    )
    return evaluate_evaluation_spec_state(
        store=store,
        evaluation_spec_state=evaluation_spec_state,
        evaluation_cases=evaluation_cases,
        base_url=base_url,
        feature_plane_repository=feature_plane_repository,
    )


def run_walk_forward_evaluation_use_case(
    *,
    store: EvaluationStore,
    evaluation_spec_id: str,
    strategy_ids: tuple[str, ...] | None,
    base_url: str,
    evaluation_cases: tuple[EvaluationCase, ...] | None = None,
):
    store.ensure_schema()
    feature_plane_repository = FeaturePlaneRepository(
        observation_repository=ObservationFrameRepository(store=store)
    )
    evaluation_spec_state = store.get_evaluation_spec(evaluation_spec_id)
    if evaluation_spec_state is None:
        raise ValueError(f"evaluation spec does not exist: {evaluation_spec_id}")
    if evaluation_cases is None:
        if not strategy_ids:
            strategy_ids = _all_strategy_ids(store)
        evaluation_cases = _evaluation_cases_for_strategy_ids(
            evaluation_spec_id=evaluation_spec_state.evaluation_spec_id,
            strategy_ids=strategy_ids,
        )
    return evaluate_evaluation_spec_state(
        store=store,
        evaluation_spec_state=evaluation_spec_state,
        evaluation_cases=evaluation_cases,
        base_url=base_url,
        feature_plane_repository=feature_plane_repository,
    )
