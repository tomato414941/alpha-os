from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionDecision:
    next_status: str
    reason: str


def decide_operator_transition(
    *,
    current_status: str,
    action: str,
) -> TransitionDecision:
    if action == "deactivate":
        if current_status != "active":
            raise ValueError(
                f"invalid hypothesis transition: {current_status} -> inactive"
            )
        return TransitionDecision(
            next_status="inactive",
            reason="operator_deactivate",
        )

    if action == "activate":
        if current_status != "inactive":
            raise ValueError(
                f"invalid hypothesis transition: {current_status} -> active"
            )
        return TransitionDecision(next_status="active", reason="operator_activate")

    raise ValueError(f"unknown transition action: {action}")
