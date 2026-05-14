from __future__ import annotations

from dataclasses import dataclass

from .data_repositories import (
    EvaluationInputRepository,
    FeaturePlaneRepository,
    ObservationFrameRepository,
)
from .evaluation_task import EvaluationTask
from .evaluation_spec import EvaluationSpec
from .evaluation_task_resolution import resolve_evaluation_tasks_for_spec
from .evaluation_report_service import resolve_report_strategy_context
from .evaluation_runner import EvaluationRunRequest, evaluate_evaluation_spec_state
from .strategy_checkpoint import StrategyCheckpoint
from .signal_discovery_application import (
    build_strategy_checkpoint_id,
    compress_screening_result_state,
    ensure_subject_set_backend_available,
    prune_screened_snapshots,
)
from .signal_discovery_execution import build_signal_discovery_execution_plan
from .signal_discovery_persistence_builders import build_strategy_checkpoint
from .signal_discovery_screening_service import screen_signal_discovery
from .store import EvaluationStore
from .subject_set_backfill_service import run_subject_set_backfill


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
class PrepareStrategyCheckpointsForEvaluationRequest:
    store: EvaluationStore
    default_target_id: str
    evaluation_spec: EvaluationSpec
    evaluation_tasks: tuple[EvaluationTask, ...]
    base_url: str
    min_sample_count: int | None
    min_abs_corr: float | None
    min_stability_score: float | None
    max_family_survivors_per_subject: int | None
    created_at: str
    feature_plane_repository: FeaturePlaneRepository
    evaluation_input_repository: EvaluationInputRepository


@dataclass(frozen=True)
class SignalDiscoveryEvaluationGroup:
    signal_discovery_id: str | None
    base_url: str
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


def group_evaluation_tasks_by_signal_discovery(
    store: EvaluationStore,
    evaluation_tasks: tuple[EvaluationTask, ...],
    *,
    base_url: str,
) -> tuple[SignalDiscoveryEvaluationGroup, ...]:
    grouped: dict[str | None, list[EvaluationTask]] = {}
    for evaluation_task in evaluation_tasks:
        strategy_state = store.get_trading_strategy(evaluation_task.strategy_id)
        if strategy_state is None:
            raise ValueError(f"evaluation task strategy does not exist: {evaluation_task.strategy_id}")
        signal_discovery_id = strategy_state.trading_strategy.signal_discovery_id
        grouped.setdefault(signal_discovery_id, []).append(evaluation_task)
    groups: list[SignalDiscoveryEvaluationGroup] = []
    for signal_discovery_id, grouped_cases in sorted(
        grouped.items(),
        key=lambda item: "" if item[0] is None else item[0],
    ):
        groups.append(
            SignalDiscoveryEvaluationGroup(
                signal_discovery_id=signal_discovery_id,
                base_url=base_url,
                evaluation_tasks=tuple(grouped_cases),
            )
        )
    return tuple(groups)


def _has_complete_strategy_checkpoints_for_fold(
    store: EvaluationStore,
    *,
    group: SignalDiscoveryEvaluationGroup,
    fold,
) -> bool:
    for evaluation_task in group.evaluation_tasks:
        strategy_checkpoints = store.list_strategy_checkpoints(
            strategy_id=evaluation_task.strategy_id,
            signal_discovery_id=group.signal_discovery_id,
            fold_label=fold.label,
            execution_start_date=fold.execution_range.start_date,
            execution_end_date=fold.execution_range.end_date,
            limit=1,
        )
        if not strategy_checkpoints:
            return False
    return True


def _backfill_strategy_checkpoints_for_fold_from_signal_discovery(
    store: EvaluationStore,
    *,
    group: SignalDiscoveryEvaluationGroup,
    fold,
    created_at: str,
) -> bool:
    shared_strategy_checkpoints = store.list_strategy_checkpoints(
        signal_discovery_id=group.signal_discovery_id,
        fold_label=fold.label,
        execution_start_date=fold.execution_range.start_date,
        execution_end_date=fold.execution_range.end_date,
        limit=1,
    )
    if not shared_strategy_checkpoints:
        return False
    source_state = shared_strategy_checkpoints[0].state
    created_any = False
    for evaluation_task in group.evaluation_tasks:
        existing_states = store.list_strategy_checkpoints(
            strategy_id=evaluation_task.strategy_id,
            signal_discovery_id=group.signal_discovery_id,
            fold_label=fold.label,
            execution_start_date=fold.execution_range.start_date,
            execution_end_date=fold.execution_range.end_date,
            limit=1,
        )
        if existing_states:
            continue
        store.upsert_strategy_checkpoint(
            state=StrategyCheckpoint(
                strategy_checkpoint_id=build_strategy_checkpoint_id(
                    strategy_id=evaluation_task.strategy_id,
                    fold_label=fold.label,
                    start_date=fold.execution_range.start_date,
                    end_date=fold.execution_range.end_date,
                ),
                strategy_id=evaluation_task.strategy_id,
                signal_discovery_id=source_state.signal_discovery_id,
                subject_set_id=source_state.subject_set_id,
                target_id=source_state.target_id,
                fold_label=source_state.fold_label,
                execution_start_date=source_state.execution_start_date,
                execution_end_date=source_state.execution_end_date,
                snapshot_set_id=source_state.snapshot_set_id,
                screening_result_id=source_state.screening_result_id,
                compressed_belief_id=source_state.compressed_belief_id,
                survivor_signal_ids=source_state.survivor_signal_ids,
                created_at=created_at,
            )
        )
        created_any = True
    return created_any


def prepare_strategy_checkpoints_for_evaluation(
    request: PrepareStrategyCheckpointsForEvaluationRequest,
) -> None:
    store = request.store
    groups = group_evaluation_tasks_by_signal_discovery(
        store,
        request.evaluation_tasks,
        base_url=request.base_url,
    )
    for group in groups:
        if group.signal_discovery_id is None:
            continue
        for fold in request.evaluation_spec.resolved_evaluation_folds:
            if _has_complete_strategy_checkpoints_for_fold(
                store,
                group=group,
                fold=fold,
            ):
                continue
            if _backfill_strategy_checkpoints_for_fold_from_signal_discovery(
                store,
                group=group,
                fold=fold,
                created_at=request.created_at,
            ):
                continue
            timestamp = request.created_at
            snapshot_set_id = (
                f"snapshot-set:{group.signal_discovery_id}:"
                f"{fold.execution_range.start_date}:"
                f"{fold.execution_range.end_date}:{timestamp}"
            )
            plan = build_signal_discovery_execution_plan(
                store,
                signal_discovery_id=group.signal_discovery_id,
                default_target_id=request.default_target_id,
            )
            signal_discovery = plan.signal_discovery
            subject_set = plan.subject_set
            ensure_subject_set_backend_available(
                subject_set,
                base_url=group.base_url,
            )
            backfill_result = run_subject_set_backfill(
                store,
                subject_set=subject_set,
                subject_set_id=signal_discovery.subject_set_id,
                signal_spec_ids=list(plan.signal_spec_ids),
                target_id=plan.target_id,
                start_date=fold.execution_range.start_date,
                end_date=fold.execution_range.end_date,
                base_url=group.base_url,
                pre_screen_top_k_per_kind=(
                    signal_discovery.selection_policy.pre_screen_top_k_per_kind
                ),
                pre_screen_min_abs_corr=(
                    signal_discovery.selection_policy.pre_screen_min_abs_corr
                ),
                probe_max_dates=signal_discovery.selection_policy.probe_max_dates,
                probe_min_sample_count=(
                    signal_discovery.selection_policy.probe_min_sample_count
                ),
                probe_min_abs_corr=signal_discovery.selection_policy.probe_min_abs_corr,
                probe_max_family_survivors_per_subject=(
                    signal_discovery.selection_policy.probe_max_family_survivors_per_subject
                ),
                survivor_min_sample_count=(
                    signal_discovery.selection_policy.survivor_min_sample_count
                ),
                survivor_min_abs_corr=(
                    signal_discovery.selection_policy.survivor_min_abs_corr
                ),
                survivor_max_family_survivors_per_subject=(
                    signal_discovery.selection_policy.survivor_max_family_survivors_per_subject
                ),
                family_ids_by_signal_spec_id=plan.family_ids_by_signal_spec_id,
                signal_discovery_id=signal_discovery.signal_discovery_id,
                feature_plane_repository=request.feature_plane_repository,
                evaluation_input_repository=request.evaluation_input_repository,
            )
            screening_state = screen_signal_discovery(
                store,
                signal_discovery_id=signal_discovery.signal_discovery_id,
                min_sample_count=request.min_sample_count,
                min_abs_corr=request.min_abs_corr,
                min_stability_score=request.min_stability_score,
                max_family_survivors_per_subject=(
                    request.max_family_survivors_per_subject
                ),
            )
            store.archive_prepared_evaluation_snapshots(
                snapshot_set_id=snapshot_set_id,
                signal_ids=[item.signal_id for item in screening_state.result.survivors],
            )
            prune_screened_snapshots(
                store,
                selected_signal_ids=backfill_result.selected_signal_ids,
                screening_state=screening_state,
                snapshot_retention=signal_discovery.selection_policy.snapshot_retention,
            )
            compressed_belief_state = compress_screening_result_state(
                store,
                screening_result_id=screening_state.screening_result_id,
            )
            for evaluation_task in group.evaluation_tasks:
                store.upsert_strategy_checkpoint(
                    state=build_strategy_checkpoint(
                        strategy_id=evaluation_task.strategy_id,
                        signal_discovery_id=signal_discovery.signal_discovery_id,
                        subject_set_id=str(subject_set.subject_set_id),
                        target_id=plan.target_id,
                        fold_label=fold.label,
                        start_date=fold.execution_range.start_date,
                        end_date=fold.execution_range.end_date,
                        snapshot_set_id=snapshot_set_id,
                        screening_state=screening_state,
                        compressed_belief_state=compressed_belief_state,
                        created_at=timestamp,
                    )
                )


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
    prepare_strategy_checkpoints_for_evaluation(
        PrepareStrategyCheckpointsForEvaluationRequest(
            store=store,
            default_target_id=request.default_target_id,
            evaluation_spec=evaluation_spec,
            evaluation_tasks=evaluation_tasks,
            base_url=request.base_url,
            min_sample_count=request.min_sample_count,
            min_abs_corr=request.min_abs_corr,
            min_stability_score=request.min_stability_score,
            max_family_survivors_per_subject=(
                request.max_family_survivors_per_subject
            ),
            created_at=request.created_at,
            feature_plane_repository=feature_plane_repository,
            evaluation_input_repository=evaluation_input_repository,
        )
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
