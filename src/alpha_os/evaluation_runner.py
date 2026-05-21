from __future__ import annotations

from .data_repositories import FeaturePlaneRepository
from .evaluation_execution_strategy import (
    EvaluationExecutionContext,
    run_strategy_evaluation,
)
from .evaluation_task import EvaluationTask
from .evaluation_report import EvaluationReport
from .evaluation_spec import build_oos_contract_summary
from .store import EvaluationStore, _utc_now


def evaluate_evaluation_spec_state(
    *,
    store: EvaluationStore,
    evaluation_spec_state: object,
    evaluation_tasks: tuple[EvaluationTask, ...],
    base_url: str,
    feature_plane_repository: FeaturePlaneRepository | None = None,
):
    evaluation_spec = evaluation_spec_state.definition
    task_results = []
    timestamp = _utc_now()
    execution_context = EvaluationExecutionContext(
        store=store,
        feature_plane_repository=feature_plane_repository,
    )
    for evaluation_task in evaluation_tasks:
        for fold in evaluation_spec.resolved_evaluation_folds:
            task_results.append(
                run_strategy_evaluation(
                    evaluation_task=evaluation_task,
                    evaluation_date_ranges=fold.resolved_evaluation_date_ranges,
                    metric_group_names=evaluation_spec.metric_group_names,
                    base_url=base_url,
                    context=execution_context,
                )
            )
    report = EvaluationReport(
        evaluation_report_id=f"{evaluation_spec_state.evaluation_spec_id}:{timestamp}",
        evaluation_spec_id=evaluation_spec_state.evaluation_spec_id,
        task_results=tuple(task_results),
        created_at=timestamp,
        oos_contract_summary=build_oos_contract_summary(evaluation_spec),
    )
    return store.upsert_evaluation_report(report=report)
