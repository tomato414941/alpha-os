from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .evaluation_lane import normalize_evaluation_lane
from .evaluation_result import EvaluationTaskResult


@dataclass(frozen=True)
class EvaluationReport:
    evaluation_report_id: str
    evaluation_spec_id: str
    task_results: tuple[EvaluationTaskResult, ...]
    created_at: str
    evaluation_lane: str = "backtest_oos"
    oos_contract_summary: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation_lane",
            normalize_evaluation_lane(self.evaluation_lane),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "evaluation_spec_id": self.evaluation_spec_id,
            "evaluation_lane": self.evaluation_lane,
            "oos_contract_summary": dict(self.oos_contract_summary),
            "task_results": [item.to_document() for item in self.task_results],
            "created_at": self.created_at,
        }

    @classmethod
    def from_document(
        cls,
        *,
        evaluation_report_id: str,
        document: dict[str, Any],
    ) -> "EvaluationReport":
        evaluation_spec_id = document.get("evaluation_spec_id")
        task_results = document.get("task_results", [])
        created_at = document.get("created_at")
        evaluation_lane = document.get("evaluation_lane")
        oos_contract_summary = document.get("oos_contract_summary", {})
        if "summaries" in document:
            raise ValueError(
                "evaluation report summaries field is no longer supported; "
                "use task_results"
            )
        if not isinstance(evaluation_spec_id, str) or not evaluation_spec_id:
            raise ValueError("evaluation report is missing evaluation_spec_id")
        if not isinstance(task_results, list):
            raise ValueError("evaluation report task_results are invalid")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("evaluation report is missing created_at")
        if not isinstance(oos_contract_summary, dict):
            raise ValueError("evaluation report oos_contract_summary is invalid")
        return cls(
            evaluation_report_id=evaluation_report_id,
            evaluation_spec_id=evaluation_spec_id,
            task_results=tuple(
                EvaluationTaskResult.from_document(item)
                for item in task_results
                if isinstance(item, dict)
            ),
            created_at=created_at,
            evaluation_lane=None if evaluation_lane is None else str(evaluation_lane),
            oos_contract_summary={
                str(key): str(value) for key, value in oos_contract_summary.items()
            },
        )
