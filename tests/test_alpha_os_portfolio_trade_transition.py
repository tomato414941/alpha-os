from __future__ import annotations

import pytest


def _target(subject_id: str, weight: float):
    from alpha_os.portfolio_decision import PortfolioTarget

    return PortfolioTarget(
        subject_id=subject_id,
        target_weight=weight,
        position_delta=0.0,
        target_notional=weight,
    )


def test_trade_transition_scales_deltas_to_turnover_budget():
    from alpha_os.portfolio_trade_transition import (
        TradeTransitionRequest,
        apply_trade_transition,
    )

    result = apply_trade_transition(
        TradeTransitionRequest(
            desired_targets={
                "A": _target("A", 0.30),
                "B": _target("B", -0.30),
            },
            current_weights={"A": 0.0, "B": 0.0},
            capital_base=1.0,
            turnover_budget=0.30,
        )
    )

    assert result.trace.desired_turnover == pytest.approx(0.60)
    assert result.trace.executed_turnover == pytest.approx(0.30)
    assert result.executed_targets["A"].target_weight == pytest.approx(0.15)
    assert result.executed_targets["B"].target_weight == pytest.approx(-0.15)
