from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .evaluation_lane import normalize_evaluation_lane
from .evaluation_result import EvaluationResult


@dataclass(frozen=True)
class EvaluationRunResult:
    evaluation_run_result_id: str
    evaluation_spec_id: str
    results: dict[str, EvaluationResult]
    created_at: str
    evaluation_lane: str = "backtest_oos"
    oos_contract_summary: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("evaluation run result requires at least one result")
        object.__setattr__(
            self,
            "evaluation_lane",
            normalize_evaluation_lane(self.evaluation_lane),
        )
        object.__setattr__(self, "results", dict(self.results))

    def to_document(self) -> dict[str, Any]:
        return {
            "evaluation_spec_id": self.evaluation_spec_id,
            "evaluation_lane": self.evaluation_lane,
            "oos_contract_summary": dict(self.oos_contract_summary),
            "results": {
                result_key: result.to_document()
                for result_key, result in self.results.items()
            },
            "created_at": self.created_at,
        }

    @classmethod
    def from_document(
        cls,
        *,
        evaluation_run_result_id: str,
        document: dict[str, Any],
    ) -> "EvaluationRunResult":
        evaluation_spec_id = document.get("evaluation_spec_id")
        results = document.get("results", {})
        created_at = document.get("created_at")
        evaluation_lane = document.get("evaluation_lane")
        oos_contract_summary = document.get("oos_contract_summary", {})
        if not isinstance(evaluation_spec_id, str) or not evaluation_spec_id:
            raise ValueError("evaluation run result is missing evaluation_spec_id")
        if not isinstance(results, dict):
            raise ValueError("evaluation run result results are invalid")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("evaluation run result is missing created_at")
        if not isinstance(oos_contract_summary, dict):
            raise ValueError("evaluation run result oos_contract_summary is invalid")
        return cls(
            evaluation_run_result_id=evaluation_run_result_id,
            evaluation_spec_id=evaluation_spec_id,
            results={
                str(result_key): EvaluationResult.from_document(item)
                for result_key, item in results.items()
                if isinstance(item, dict)
            },
            created_at=created_at,
            evaluation_lane=None if evaluation_lane is None else str(evaluation_lane),
            oos_contract_summary={
                str(key): str(value) for key, value in oos_contract_summary.items()
            },
        )
