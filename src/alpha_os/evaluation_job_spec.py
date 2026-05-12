from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, init=False)
class EvaluationJobSpec:
    evaluation_task_id: str
    strategy_checkpoint_id: str | None = None

    def __init__(
        self,
        *,
        evaluation_task_id: str | None = None,
        strategy_checkpoint_id: str | None = None,
    ) -> None:
        if evaluation_task_id is None:
            raise ValueError("evaluation job spec requires evaluation_task_id")
        object.__setattr__(self, "evaluation_task_id", evaluation_task_id)
        object.__setattr__(
            self,
            "strategy_checkpoint_id",
            strategy_checkpoint_id,
        )

    def to_document(self) -> dict[str, Any]:
        document = {
            "evaluation_task_id": self.evaluation_task_id,
        }
        if self.strategy_checkpoint_id is not None:
            document["strategy_checkpoint_id"] = (
                self.strategy_checkpoint_id
            )
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "EvaluationJobSpec":
        strategy_checkpoint_id_document = document.get(
            "strategy_checkpoint_id"
        )
        evaluation_task_id = document.get("evaluation_task_id")
        if evaluation_task_id is None:
            raise ValueError("evaluation job spec is missing evaluation_task_id")
        return cls(
            evaluation_task_id=str(evaluation_task_id),
            strategy_checkpoint_id=(
                None
                if strategy_checkpoint_id_document is None
                else str(strategy_checkpoint_id_document)
            ),
        )


def default_evaluation_job_spec(
    evaluation_task_id: str | None = None,
) -> EvaluationJobSpec:
    return EvaluationJobSpec(
        evaluation_task_id=evaluation_task_id,
    )
