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
    if action == "pause":
        if current_status != "registered":
            raise ValueError(
                f"invalid hypothesis transition: {current_status} -> paused"
            )
        return TransitionDecision(next_status="paused", reason="operator_pause")

    if action == "resume":
        if current_status != "paused":
            raise ValueError(
                f"invalid hypothesis transition: {current_status} -> registered"
            )
        return TransitionDecision(next_status="registered", reason="operator_resume")

    if action == "retire":
        if current_status not in {"registered", "paused"}:
            raise ValueError(
                f"invalid hypothesis transition: {current_status} -> retired"
            )
        return TransitionDecision(next_status="retired", reason="operator_retire")

    raise ValueError(f"unknown transition action: {action}")
