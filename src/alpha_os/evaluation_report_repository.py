from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evaluation_report import EvaluationReport
from .store import EvaluationStore


@dataclass(frozen=True, init=False)
class PendingEvaluationDecisionTrace:
    evaluation_task_id: str
    evaluation_fold_label: str
    evaluation_range_label: str
    result: object
    subject_metadata_by_subject: dict[str, dict[str, str]]

    def __init__(
        self,
        *,
        evaluation_task_id: str | None = None,
        evaluation_fold_label: str,
        evaluation_range_label: str,
        result: object,
        subject_metadata_by_subject: dict[str, dict[str, str]] | None = None,
    ) -> None:
        if evaluation_task_id is None:
            raise ValueError("pending decision trace requires evaluation_task_id")
        object.__setattr__(self, "evaluation_task_id", evaluation_task_id)
        object.__setattr__(self, "evaluation_fold_label", evaluation_fold_label)
        object.__setattr__(self, "evaluation_range_label", evaluation_range_label)
        object.__setattr__(self, "result", result)
        object.__setattr__(
            self,
            "subject_metadata_by_subject",
            subject_metadata_by_subject or {},
        )


@dataclass(frozen=True)
class EvaluationReportRepository:
    store: EvaluationStore

    def upsert_report(self, *, report: EvaluationReport):
        return self.store.upsert_evaluation_report(report=report)

    def upsert_decision_trace(
        self,
        *,
        evaluation_report_id: str,
        trace: PendingEvaluationDecisionTrace,
        variant: str = "selected",
        step_granularity: str = "1d",
    ) -> None:
        self.store.upsert_evaluation_decision_trace(
            evaluation_report_id=evaluation_report_id,
            evaluation_task_id=trace.evaluation_task_id,
            evaluation_fold_label=trace.evaluation_fold_label,
            evaluation_range_label=trace.evaluation_range_label,
            result=trace.result,
            variant=variant,
            step_granularity=step_granularity,
            subject_metadata_by_subject=trace.subject_metadata_by_subject,
        )

    def upsert_report_with_traces(
        self,
        *,
        report: EvaluationReport,
        pending_decision_traces: tuple[PendingEvaluationDecisionTrace, ...],
    ):
        report_state = self.upsert_report(report=report)
        for trace in pending_decision_traces:
            self.upsert_decision_trace(
                evaluation_report_id=report.evaluation_report_id,
                trace=trace,
            )
        return report_state

    def get_report(self, evaluation_report_id: str):
        return self.store.get_evaluation_report(evaluation_report_id)

    def list_reports(
        self,
        *,
        evaluation_spec_id: str | None = None,
        limit: int = 20,
    ):
        return self.store.list_evaluation_reports(
            evaluation_spec_id=evaluation_spec_id,
            limit=limit,
        )

    def list_decision_trace_steps(self, **filters: Any):
        return self.store.list_evaluation_decision_trace_steps(**filters)

    def list_decision_trace_subject_steps(self, **filters: Any):
        return self.store.list_evaluation_decision_trace_subject_steps(**filters)
