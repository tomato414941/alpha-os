from __future__ import annotations

import time
from dataclasses import dataclass

from .data_repositories import (
    EvaluationInputRepository,
    FeaturePlaneRepository,
    ObservationFrameRepository,
)
from .evaluation_task import EvaluationTask
from .strategy_training import build_signal_train_id
from .evaluation_task_resolution import resolve_evaluation_tasks_for_spec
from .evaluation_report_service import resolve_report_strategy_context
from .evaluation_runner import EvaluationRunRequest, evaluate_evaluation_spec_state
from .initial_strategy_state import InitialStrategyState
from .signal_discovery_application import (
    build_initial_strategy_state_id,
    build_prepared_evaluation_snapshot_set_id,
    build_signal_discovery_run_id,
    persist_initial_strategy_state,
    persist_signal_discovery_run,
    run_signal_discovery_workflow,
)
from .store import EvaluationStore


@dataclass(frozen=True)
class RunEvaluationUseCaseRequest:
    store: EvaluationStore
    default_target_id: str
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
    report_subject_set_context: dict[str, str]


@dataclass(frozen=True)
class RunWalkForwardEvaluationUseCaseRequest:
    store: EvaluationStore
    default_target_id: str
    evaluation_spec_id: str
    sizing_method: str | None
    sizing_engine: str | None
    direction_mode: str | None
    strategy_ids: tuple[str, ...] | None
    evaluation_task_ids: tuple[str, ...] | None
    base_url: str
    min_sample_count: int | None
    min_abs_corr: float | None
    min_stability_score: float | None
    max_family_survivors_per_subject: int | None
    created_at: str


@dataclass(frozen=True)
class RunWalkForwardEvaluationUseCaseResult:
    report_state: object


@dataclass(frozen=True)
class SignalTrainGroup:
    signal_train_id: str
    signal_discovery_id: str | None
    base_url: str
    requires_signal_train: bool
    evaluation_tasks: tuple[EvaluationTask, ...]


def run_evaluation_use_case(
    request: RunEvaluationUseCaseRequest,
) -> RunEvaluationUseCaseResult:
    store = request.store
    store.ensure_schema()
    feature_plane_repository = FeaturePlaneRepository(
        observation_repository=ObservationFrameRepository(store=store)
    )
    evaluation_input_repository = EvaluationInputRepository()
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
            default_target_id=request.default_target_id,
            evaluation_spec_state=evaluation_spec_state,
            evaluation_tasks=evaluation_tasks,
            base_url=request.base_url,
            feature_plane_repository=feature_plane_repository,
            evaluation_input_repository=evaluation_input_repository,
        )
    )
    return RunEvaluationUseCaseResult(
        report_state=report_state,
        report_subject_set_context=resolve_report_strategy_context(
            store,
            report_state=report_state,
        ),
    )


def group_evaluation_tasks_by_signal_train(
    store: EvaluationStore,
    evaluation_tasks: tuple[EvaluationTask, ...],
    *,
    base_url: str,
) -> tuple[SignalTrainGroup, ...]:
    grouped: dict[str, list[EvaluationTask]] = {}
    for evaluation_task in evaluation_tasks:
        strategy_state = store.get_trading_strategy(evaluation_task.strategy_id)
        if strategy_state is None:
            raise ValueError(f"evaluation task strategy does not exist: {evaluation_task.strategy_id}")
        signal_train_id = build_signal_train_id(
            signal_discovery_id=strategy_state.trading_strategy.signal_discovery_id,
        )
        grouped.setdefault(signal_train_id, []).append(evaluation_task)
    groups: list[SignalTrainGroup] = []
    for signal_train_id, grouped_cases in sorted(grouped.items()):
        requires_signal_train = False
        signal_discovery_id = None
        for case in grouped_cases:
            strategy_state = store.get_trading_strategy(case.strategy_id)
            if strategy_state is None:
                raise ValueError(f"evaluation task strategy does not exist: {case.strategy_id}")
            if signal_discovery_id is None:
                signal_discovery_id = strategy_state.trading_strategy.signal_discovery_id
            if strategy_state.trading_strategy.requires_signal_train:
                requires_signal_train = True
                break
        groups.append(
            SignalTrainGroup(
                signal_train_id=signal_train_id,
                signal_discovery_id=signal_discovery_id,
                base_url=base_url,
                requires_signal_train=requires_signal_train,
                evaluation_tasks=tuple(grouped_cases),
            )
        )
    return tuple(groups)


def _has_complete_initial_strategy_states_for_fold(
    store: EvaluationStore,
    *,
    signal_train_group: SignalTrainGroup,
    fold,
) -> bool:
    for evaluation_task in signal_train_group.evaluation_tasks:
        initial_strategy_states = store.list_initial_strategy_states(
            strategy_id=evaluation_task.strategy_id,
            signal_train_id=signal_train_group.signal_train_id,
            fold_label=fold.label,
            execution_start_date=fold.execution_range.start_date,
            execution_end_date=fold.execution_range.end_date,
            limit=1,
        )
        if not initial_strategy_states:
            return False
    return True


def _backfill_initial_strategy_states_for_fold_from_signal_train(
    store: EvaluationStore,
    *,
    signal_train_group: SignalTrainGroup,
    fold,
    created_at: str,
) -> bool:
    shared_initial_strategy_states = store.list_initial_strategy_states(
        signal_train_id=signal_train_group.signal_train_id,
        fold_label=fold.label,
        execution_start_date=fold.execution_range.start_date,
        execution_end_date=fold.execution_range.end_date,
        limit=1,
    )
    if not shared_initial_strategy_states:
        return False
    source_state = shared_initial_strategy_states[0].state
    created_any = False
    for evaluation_task in signal_train_group.evaluation_tasks:
        existing_states = store.list_initial_strategy_states(
            strategy_id=evaluation_task.strategy_id,
            signal_train_id=signal_train_group.signal_train_id,
            fold_label=fold.label,
            execution_start_date=fold.execution_range.start_date,
            execution_end_date=fold.execution_range.end_date,
            limit=1,
        )
        if existing_states:
            continue
        store.upsert_initial_strategy_state(
            state=InitialStrategyState(
                initial_strategy_state_id=build_initial_strategy_state_id(
                    strategy_id=evaluation_task.strategy_id,
                    fold_label=fold.label,
                    start_date=fold.execution_range.start_date,
                    end_date=fold.execution_range.end_date,
                ),
                strategy_id=evaluation_task.strategy_id,
                signal_train_id=signal_train_group.signal_train_id,
                signal_discovery_id=source_state.signal_discovery_id,
                subject_set_id=source_state.subject_set_id,
                target_id=source_state.target_id,
                fold_label=source_state.fold_label,
                execution_start_date=source_state.execution_start_date,
                execution_end_date=source_state.execution_end_date,
                signal_discovery_run_id=source_state.signal_discovery_run_id,
                snapshot_set_id=source_state.snapshot_set_id,
                screening_result_id=source_state.screening_result_id,
                compressed_belief_id=source_state.compressed_belief_id,
                survivor_signal_ids=source_state.survivor_signal_ids,
                created_at=created_at,
            )
        )
        created_any = True
    return created_any


def run_walk_forward_evaluation_use_case(
    request: RunWalkForwardEvaluationUseCaseRequest,
) -> RunWalkForwardEvaluationUseCaseResult:
    store = request.store
    store.ensure_schema()
    feature_plane_repository = FeaturePlaneRepository(
        observation_repository=ObservationFrameRepository(store=store)
    )
    evaluation_input_repository = EvaluationInputRepository()
    evaluation_spec_state = store.get_evaluation_spec(request.evaluation_spec_id)
    if evaluation_spec_state is None:
        raise ValueError(f"evaluation spec does not exist: {request.evaluation_spec_id}")
    evaluation_spec = evaluation_spec_state.definition
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
    signal_train_groups = group_evaluation_tasks_by_signal_train(
        store,
        evaluation_tasks,
        base_url=request.base_url,
    )
    for signal_train_group in signal_train_groups:
        if not signal_train_group.requires_signal_train:
            continue
        if signal_train_group.signal_discovery_id is None:
            raise ValueError(
                "trained signal train group is missing signal discovery provenance: "
                f"{signal_train_group.signal_train_id}"
            )
        for fold in evaluation_spec.resolved_evaluation_folds:
            if _has_complete_initial_strategy_states_for_fold(
                store,
                signal_train_group=signal_train_group,
                fold=fold,
            ):
                continue
            if _backfill_initial_strategy_states_for_fold_from_signal_train(
                store,
                signal_train_group=signal_train_group,
                fold=fold,
                created_at=request.created_at,
            ):
                continue
            timestamp = request.created_at
            started_at = time.perf_counter()
            signal_discovery_run_id = build_signal_discovery_run_id(
                signal_discovery_id=signal_train_group.signal_discovery_id,
                start_date=fold.execution_range.start_date,
                end_date=fold.execution_range.end_date,
                created_at=timestamp,
            )
            snapshot_set_id = build_prepared_evaluation_snapshot_set_id(
                signal_discovery_id=signal_train_group.signal_discovery_id,
                start_date=fold.execution_range.start_date,
                end_date=fold.execution_range.end_date,
                created_at=timestamp,
            )
            (
                backfill_result,
                signal_discovery,
                subject_set,
                target_id,
                screening_state,
                compressed_belief_state,
                pruned_snapshot_count,
            ) = run_signal_discovery_workflow(
                store,
                default_target_id=request.default_target_id,
                signal_discovery_run_id=signal_discovery_run_id,
                snapshot_set_id=snapshot_set_id,
                signal_discovery_id=signal_train_group.signal_discovery_id,
                start_date=fold.execution_range.start_date,
                end_date=fold.execution_range.end_date,
                base_url=signal_train_group.base_url,
                min_sample_count=request.min_sample_count,
                min_abs_corr=request.min_abs_corr,
                min_stability_score=request.min_stability_score,
                max_family_survivors_per_subject=(
                    request.max_family_survivors_per_subject
                ),
                feature_plane_repository=feature_plane_repository,
                evaluation_input_repository=evaluation_input_repository,
            )
            persist_signal_discovery_run(
                store,
                signal_discovery_run_id=signal_discovery_run_id,
                snapshot_set_id=snapshot_set_id,
                signal_discovery_id=signal_discovery.signal_discovery_id,
                subject_set_id=str(subject_set.subject_set_id),
                target_id=target_id,
                start_date=fold.execution_range.start_date,
                end_date=fold.execution_range.end_date,
                screening_result_id=screening_state.screening_result_id,
                compressed_belief_id=compressed_belief_state.compressed_belief_id,
                workflow_runtime_s=time.perf_counter() - started_at,
                backfill_result=backfill_result,
                pruned_snapshot_count=pruned_snapshot_count,
                created_at=timestamp,
            )
            for evaluation_task in signal_train_group.evaluation_tasks:
                persist_initial_strategy_state(
                    store,
                    strategy_id=evaluation_task.strategy_id,
                    signal_train_id=signal_train_group.signal_train_id,
                    signal_discovery_id=signal_discovery.signal_discovery_id,
                    subject_set_id=str(subject_set.subject_set_id),
                    target_id=target_id,
                    fold_label=fold.label,
                    start_date=fold.execution_range.start_date,
                    end_date=fold.execution_range.end_date,
                    signal_discovery_run_id=signal_discovery_run_id,
                    snapshot_set_id=snapshot_set_id,
                    screening_state=screening_state,
                    compressed_belief_state=compressed_belief_state,
                    created_at=timestamp,
                )
    report_state = evaluate_evaluation_spec_state(
        EvaluationRunRequest(
            store=store,
            default_target_id=request.default_target_id,
            evaluation_spec_state=evaluation_spec_state,
            evaluation_tasks=evaluation_tasks,
            base_url=request.base_url,
            feature_plane_repository=feature_plane_repository,
            evaluation_input_repository=evaluation_input_repository,
        )
    )
    return RunWalkForwardEvaluationUseCaseResult(report_state=report_state)
