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
        return TransitionDecision(
            next_status="registered",
            reason="keep_registered_with_live_label",
        )
    return TransitionDecision(
        next_status="registered",
        reason="keep_registered_without_live_label",
    )


def decide_operator_transition(
    *,
    current_status: str,
    allocation_trust: float,
    action: str,
) -> TransitionDecision:
    if action == "pause":
        if current_status != "registered" or float(allocation_trust) <= 0.0:
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
