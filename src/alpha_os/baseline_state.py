from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


BaselineStatus = Literal["active", "retired"]


@dataclass(frozen=True)
class BaselineState:
    baseline_id: str
    strategy_id: str
    source_promotion_decision_id: str | None
    active_from: str
    status: BaselineStatus

    def to_document(self) -> dict[str, object]:
        return {
            "baseline_id": self.baseline_id,
            "strategy_id": self.strategy_id,
            "source_promotion_decision_id": self.source_promotion_decision_id,
            "active_from": self.active_from,
            "status": self.status,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "BaselineState":
        baseline_id = document.get("baseline_id")
        strategy_id = document.get("strategy_id")
        source_promotion_decision_id = document.get("source_promotion_decision_id")
        active_from = document.get("active_from")
        status = document.get("status")
        if not isinstance(baseline_id, str) or not baseline_id:
            raise ValueError("baseline state is missing baseline_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise ValueError("baseline state is missing strategy_id")
        if source_promotion_decision_id is not None and not isinstance(
            source_promotion_decision_id,
            str,
        ):
            raise ValueError(
                "baseline state source_promotion_decision_id is invalid"
            )
        if not isinstance(active_from, str) or not active_from:
            raise ValueError("baseline state is missing active_from")
        if status not in ("active", "retired"):
            raise ValueError("baseline state status is invalid")
        return cls(
            baseline_id=baseline_id,
            strategy_id=strategy_id,
            source_promotion_decision_id=source_promotion_decision_id,
            active_from=active_from,
            status=status,
        )


def baseline_from_promotion_decision(
    *,
    baseline_id: str,
    strategy_id: str,
    promotion_decision,
    active_from: str,
) -> BaselineState:
    if promotion_decision.status != "promote":
        raise ValueError("baseline can only be created from promoted decision")
    return BaselineState(
        baseline_id=baseline_id,
        strategy_id=strategy_id,
        source_promotion_decision_id=promotion_decision.promotion_decision_id,
        active_from=active_from,
        status="active",
    )
