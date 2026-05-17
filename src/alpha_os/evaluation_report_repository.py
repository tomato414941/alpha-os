from __future__ import annotations

from dataclasses import dataclass

from .evaluation_report import EvaluationReport
from .store import EvaluationStore


@dataclass(frozen=True)
class EvaluationReportRepository:
    store: EvaluationStore

    def upsert_report(self, *, report: EvaluationReport):
        return self.store.upsert_evaluation_report(report=report)

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
