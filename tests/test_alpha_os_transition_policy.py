from __future__ import annotations

import pytest


def test_decide_operator_transition_allows_activate_and_deactivate():
    from alpha_os.transition_policy import decide_operator_transition

    assert (
        decide_operator_transition(
            current_status="active",
            action="deactivate",
        ).next_status
        == "inactive"
    )
    assert (
        decide_operator_transition(
            current_status="inactive",
            action="activate",
        ).next_status
        == "active"
    )


def test_decide_operator_transition_rejects_invalid_deactivate_from_inactive():
    from alpha_os.transition_policy import decide_operator_transition

    with pytest.raises(ValueError):
        decide_operator_transition(
            current_status="inactive",
            action="deactivate",
        )
