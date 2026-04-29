from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


def build_evaluation_task_id(
    *,
    strategy_id: str,
    evaluation_spec_id: str | None = None,
) -> str:
    if evaluation_spec_id is None:
        raise ValueError("evaluation task id requires evaluation_spec_id")
    payload_parts = [
        strategy_id,
        evaluation_spec_id,
    ]
    payload = "|".join(payload_parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"task:{digest}"


@dataclass(frozen=True, init=False)
class EvaluationTask:
    evaluation_task_id: str
    strategy_id: str
    evaluation_spec_id: str

    def __init__(
        self,
        *,
        evaluation_task_id: str | None = None,
        strategy_id: str,
        evaluation_spec_id: str | None = None,
    ) -> None:
        if evaluation_task_id is None:
            raise ValueError("evaluation task requires evaluation_task_id")
        if evaluation_spec_id is None:
            raise ValueError("evaluation task requires evaluation_spec_id")
        object.__setattr__(self, "evaluation_task_id", evaluation_task_id)
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "evaluation_spec_id", evaluation_spec_id)

    def to_document(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "evaluation_spec_id": self.evaluation_spec_id,
        }

    @classmethod
    def from_document(
        cls,
        *,
        evaluation_task_id: str | None = None,
        document: dict[str, Any],
    ) -> "EvaluationTask":
        if evaluation_task_id is None:
            raise ValueError("evaluation task requires evaluation_task_id")
        evaluation_spec_id = document.get("evaluation_spec_id")
        if evaluation_spec_id is None:
            raise ValueError("evaluation task is missing evaluation_spec_id")
        return cls(
            evaluation_task_id=evaluation_task_id,
            strategy_id=str(document["strategy_id"]),
            evaluation_spec_id=str(evaluation_spec_id),
        )
