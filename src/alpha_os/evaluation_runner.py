from __future__ import annotations

from .data_repositories import FeaturePlaneRepository
from .evaluation_execution_strategy import (
    EvaluationExecutionContext,
    run_strategy_evaluation,
)
from .evaluation_run_result import EvaluationRunResult
from .evaluation_spec import build_oos_contract_summary
from .store import EvaluationStore, _utc_now

EvaluationTarget = tuple[str, str]


def evaluate_evaluation_spec_state(
    *,
    store: EvaluationStore,
    evaluation_spec_state: object,
    evaluation_targets: tuple[EvaluationTarget, ...],
    base_url: str,
    feature_plane_repository: FeaturePlaneRepository | None = None,
):
    evaluation_spec = evaluation_spec_state.definition
    results = {}
    timestamp = _utc_now()
    execution_context = EvaluationExecutionContext(
        store=store,
        feature_plane_repository=feature_plane_repository,
    )
    folds = tuple(evaluation_spec.resolved_evaluation_folds)
    for result_key, strategy_id in evaluation_targets:
        for fold in folds:
            effective_result_key = (
                result_key if len(folds) == 1 else f"{result_key}:{fold.label}"
            )
            results[effective_result_key] = run_strategy_evaluation(
                strategy_id=strategy_id,
                evaluation_date_ranges=fold.resolved_evaluation_date_ranges,
                metric_group_names=evaluation_spec.metric_group_names,
                base_url=base_url,
                context=execution_context,
            )
    run_result = EvaluationRunResult(
        evaluation_run_result_id=f"{evaluation_spec_state.evaluation_spec_id}:{timestamp}",
        evaluation_spec_id=evaluation_spec_state.evaluation_spec_id,
        results=results,
        created_at=timestamp,
        oos_contract_summary=build_oos_contract_summary(evaluation_spec),
    )
    return store.upsert_evaluation_run_result(run_result=run_result)
