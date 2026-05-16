from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .data_repositories import EvaluationInputRepository, FeaturePlaneRepository
from .evaluation_execution_strategy import (
    EvaluationExecutionContext,
    evaluation_execution_strategy_for_request,
    frozen_snapshot_start_date as frozen_snapshot_start_date,
    frozen_survivor_definitions as frozen_survivor_definitions,
    generate_frozen_survivor_test_snapshots as generate_frozen_survivor_test_snapshots,
    requires_frozen_test_application as requires_frozen_test_application,
    strategy_sleeve_attribution_summaries as strategy_sleeve_attribution_summaries,
    subject_matches_sleeve_filter as subject_matches_sleeve_filter,
)
from .evaluation_task import EvaluationTask
from .evaluation_plan import build_evaluation_plan
from .evaluation_report import EvaluationReport
from .evaluation_report_repository import (
    EvaluationReportRepository,
    PendingEvaluationDecisionTrace,
)
from .evaluation_spec import build_oos_contract_summary
from .store import EvaluationStore, _utc_now


class EvaluationReportWriter(Protocol):
    def upsert_report_with_traces(
        self,
        *,
        report: EvaluationReport,
        pending_decision_traces: tuple[PendingEvaluationDecisionTrace, ...],
    ):
        ...


@dataclass(frozen=True, init=False)
class EvaluationRunRequest:
    store: EvaluationStore
    default_target_id: str
    evaluation_spec_state: object
    evaluation_tasks: tuple[EvaluationTask, ...]
    base_url: str
    feature_plane_repository: FeaturePlaneRepository | None = None
    evaluation_input_repository: EvaluationInputRepository | None = None
    report_writer: EvaluationReportWriter | None = None

    def __init__(
        self,
        *,
        store: EvaluationStore,
        default_target_id: str,
        evaluation_spec_state: object,
        evaluation_tasks: tuple[EvaluationTask, ...] | None = None,
        base_url: str,
        feature_plane_repository: FeaturePlaneRepository | None = None,
        evaluation_input_repository: EvaluationInputRepository | None = None,
        report_writer: EvaluationReportWriter | None = None,
    ) -> None:
        if evaluation_tasks is None:
            raise ValueError("evaluation run request requires evaluation_tasks")
        object.__setattr__(self, "store", store)
        object.__setattr__(self, "default_target_id", default_target_id)
        object.__setattr__(self, "evaluation_spec_state", evaluation_spec_state)
        object.__setattr__(self, "evaluation_tasks", evaluation_tasks)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "feature_plane_repository", feature_plane_repository)
        object.__setattr__(
            self,
            "evaluation_input_repository",
            evaluation_input_repository,
        )
        object.__setattr__(self, "report_writer", report_writer)


def persist_evaluation_report_with_traces(
    report_writer: EvaluationReportWriter,
    *,
    report: EvaluationReport,
    pending_decision_traces: tuple[PendingEvaluationDecisionTrace, ...],
):
    return report_writer.upsert_report_with_traces(
        report=report,
        pending_decision_traces=pending_decision_traces,
    )


def evaluate_evaluation_spec_state(request: EvaluationRunRequest):
    store = request.store
    evaluation_spec_state = request.evaluation_spec_state
    evaluation_spec = evaluation_spec_state.definition
    report_writer = (
        EvaluationReportRepository(store)
        if request.report_writer is None
        else request.report_writer
    )
    evaluation_plan = build_evaluation_plan(
        store,
        evaluation_spec_id=evaluation_spec_state.evaluation_spec_id,
        evaluation_spec=evaluation_spec,
        evaluation_tasks=request.evaluation_tasks,
        default_target_id=request.default_target_id,
        base_url=request.base_url,
    )
    task_results = []
    pending_decision_traces: list[PendingEvaluationDecisionTrace] = []
    timestamp = _utc_now()
    execution_context = EvaluationExecutionContext(
        store=store,
        evaluation_spec=evaluation_spec,
        feature_plane_repository=request.feature_plane_repository,
        evaluation_input_repository=request.evaluation_input_repository,
    )
    for execution_request in evaluation_plan.execution_requests:
        execution_result = evaluation_execution_strategy_for_request(execution_request).run(
            execution_request=execution_request,
            context=execution_context,
        )
        task_results.append(execution_result.task_result)
        pending_decision_traces.extend(execution_result.pending_decision_traces)
    report = EvaluationReport(
        evaluation_report_id=f"{evaluation_spec_state.evaluation_spec_id}:{timestamp}",
        evaluation_spec_id=evaluation_spec_state.evaluation_spec_id,
        task_results=tuple(task_results),
        created_at=timestamp,
        oos_contract_summary=build_oos_contract_summary(evaluation_spec),
    )
    return persist_evaluation_report_with_traces(
        report_writer,
        report=report,
        pending_decision_traces=tuple(pending_decision_traces),
    )
