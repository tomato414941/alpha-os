from __future__ import annotations

import pytest


def test_decide_status_after_update_promotes_to_live_when_trust_is_positive():
    from alpha_os.transition_policy import decide_status_after_update

    decision = decide_status_after_update(
        current_status="registered",
        allocation_trust_after=0.01,
    )

    assert decision.next_status == "live"
    assert decision.reason == "promote_to_live"


def test_decide_status_after_update_keeps_active_when_trust_is_zero():
    from alpha_os.transition_policy import decide_status_after_update

    decision = decide_status_after_update(
        current_status="active",
        allocation_trust_after=0.0,
    )

    assert decision.next_status == "active"
    assert decision.reason == "keep_active"


def test_decide_status_after_update_rejects_paused_hypothesis():
    from alpha_os.transition_policy import decide_status_after_update

    with pytest.raises(ValueError):
        decide_status_after_update(
            current_status="paused",
            allocation_trust_after=0.1,
        )


def test_decide_operator_transition_allows_pause_resume_and_retire():
    from alpha_os.transition_policy import decide_operator_transition

    assert decide_operator_transition(current_status="live", action="pause").next_status == "paused"
    assert (
        decide_operator_transition(current_status="paused", action="resume").next_status
        == "active"
    )
    assert (
        decide_operator_transition(current_status="active", action="retire").next_status
        == "retired"
    )


def test_decide_operator_transition_rejects_invalid_transition():
    from alpha_os.transition_policy import decide_operator_transition

    with pytest.raises(ValueError):
        decide_operator_transition(current_status="registered", action="pause")
