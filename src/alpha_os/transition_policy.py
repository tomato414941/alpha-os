from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionDecision:
    next_status: str
    reason: str


def decide_status_after_update(
    *,
    current_status: str,
    allocation_trust_after: float,
) -> TransitionDecision:
    if current_status in {"paused", "retired"}:
        raise ValueError(
            f"state cannot be updated while hypothesis is {current_status}"
        )

    trust_positive = float(allocation_trust_after) > 0.0
    if trust_positive:
        if current_status == "live":
            return TransitionDecision(next_status="live", reason="keep_live")
        return TransitionDecision(next_status="live", reason="promote_to_live")

    if current_status == "registered":
        return TransitionDecision(next_status="active", reason="activate_without_allocation")
    if current_status == "active":
        return TransitionDecision(next_status="active", reason="keep_active")
    return TransitionDecision(next_status="active", reason="demote_to_active")


def decide_operator_transition(*, current_status: str, action: str) -> TransitionDecision:
    if action == "pause":
        if current_status != "live":
            raise ValueError(
                f"invalid hypothesis transition: {current_status} -> paused"
            )
        return TransitionDecision(next_status="paused", reason="operator_pause")

    if action == "resume":
        if current_status != "paused":
            raise ValueError(
                f"invalid hypothesis transition: {current_status} -> active"
            )
        return TransitionDecision(next_status="active", reason="operator_resume")

    if action == "retire":
        if current_status not in {"active", "paused"}:
            raise ValueError(
                f"invalid hypothesis transition: {current_status} -> retired"
            )
        return TransitionDecision(next_status="retired", reason="operator_retire")

    raise ValueError(f"unknown transition action: {action}")
