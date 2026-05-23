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


def test_execution_policy_holds_small_delta_inside_no_trade_band():
    from alpha_os.portfolio_execution_policy import (
        ExecutionPolicySpec,
        TradeTransitionRequest,
        apply_execution_policy,
    )

    result = apply_execution_policy(
        TradeTransitionRequest(
            desired_targets={"A": _target("A", 0.103)},
            current_weights={"A": 0.10},
            capital_base=1.0,
            execution_policy=ExecutionPolicySpec(no_trade_band=0.005),
        )
    )

    assert result.executed_targets["A"].target_weight == pytest.approx(0.10)
    assert result.trace.desired_turnover == pytest.approx(0.003)
    assert result.trace.executed_turnover == pytest.approx(0.0)
    assert result.trace.skipped_trade_count == 1


def test_execution_policy_scales_deltas_to_turnover_budget():
    from alpha_os.portfolio_execution_policy import (
        ExecutionPolicySpec,
        TradeTransitionRequest,
        apply_execution_policy,
    )

    result = apply_execution_policy(
        TradeTransitionRequest(
            desired_targets={
                "A": _target("A", 0.30),
                "B": _target("B", -0.30),
            },
            current_weights={"A": 0.0, "B": 0.0},
            capital_base=1.0,
            execution_policy=ExecutionPolicySpec(turnover_budget=0.30),
        )
    )

    assert result.trace.desired_turnover == pytest.approx(0.60)
    assert result.trace.executed_turnover == pytest.approx(0.30)
    assert result.executed_targets["A"].target_weight == pytest.approx(0.15)
    assert result.executed_targets["B"].target_weight == pytest.approx(-0.15)


def test_execution_policy_soft_thresholds_turnover_cost_for_existing_positions():
    from alpha_os.portfolio_execution_policy import (
        ExecutionPolicySpec,
        TradeTransitionRequest,
        apply_execution_policy,
    )

    result = apply_execution_policy(
        TradeTransitionRequest(
            desired_targets={
                "A": _target("A", 0.105),
                "B": _target("B", 0.14),
            },
            current_weights={"A": 0.10, "B": 0.10},
            capital_base=10.0,
            execution_policy=ExecutionPolicySpec(transition_soft_threshold=0.02),
            per_turnover_cost=0.01,
        )
    )

    assert result.executed_targets["A"].target_weight == pytest.approx(0.10)
    assert result.executed_targets["B"].target_weight == pytest.approx(0.12)
    assert result.trace.skipped_trade_count == 1
    assert result.trace.expected_execution_cost == pytest.approx(0.002)
