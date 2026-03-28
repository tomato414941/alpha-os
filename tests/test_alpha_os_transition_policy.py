from __future__ import annotations

import pytest


def test_decide_operator_transition_allows_pause_resume_and_retire():
    from alpha_os.transition_policy import decide_operator_transition

    assert (
        decide_operator_transition(
            current_status="registered",
            action="pause",
        ).next_status
        == "paused"
    )
    assert (
        decide_operator_transition(
            current_status="paused",
            action="resume",
        ).next_status
        == "registered"
    )
    assert (
        decide_operator_transition(
            current_status="registered",
            action="retire",
        ).next_status
        == "retired"
    )


def test_decide_operator_transition_rejects_invalid_pause_from_paused():
    from alpha_os.transition_policy import decide_operator_transition

    with pytest.raises(ValueError):
        decide_operator_transition(
            current_status="paused",
            action="pause",
        )
