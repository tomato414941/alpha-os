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


def test_threshold_execution_policy_scales_deltas_to_turnover_budget():
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
            execution_policy=ExecutionPolicySpec(mode="threshold", turnover_budget=0.30),
        )
    )

    assert result.trace.desired_turnover == pytest.approx(0.60)
    assert result.trace.executed_turnover == pytest.approx(0.30)
    assert result.executed_targets["A"].target_weight == pytest.approx(0.15)
    assert result.executed_targets["B"].target_weight == pytest.approx(-0.15)


def test_utility_execution_policy_prioritizes_high_utility_turnover():
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
            signal_value_by_subject={"A": 0.1, "B": -1.0},
        )
    )

    assert result.trace.desired_turnover == pytest.approx(0.60)
    assert result.trace.executed_turnover == pytest.approx(0.30)
    assert result.executed_targets["A"].target_weight == pytest.approx(0.0)
    assert result.executed_targets["B"].target_weight == pytest.approx(-0.30)
    assert result.trace.priority_filled_turnover == pytest.approx(0.30)


def test_execution_policy_soft_thresholds_costly_small_trades():
    from alpha_os.portfolio_execution_policy import (
        ExecutionPolicySpec,
        TradeTransitionRequest,
        apply_execution_policy,
    )

    result = apply_execution_policy(
        TradeTransitionRequest(
            desired_targets={
                "A": _target("A", 0.01),
                "B": _target("B", -0.05),
            },
            current_weights={"A": 0.0, "B": 0.0},
            capital_base=10.0,
            execution_policy=ExecutionPolicySpec(cost_soft_threshold=0.02),
            per_turnover_cost=0.01,
        )
    )

    assert result.executed_targets["A"].target_weight == pytest.approx(0.0)
    assert result.executed_targets["B"].target_weight == pytest.approx(-0.03)
    assert result.trace.skipped_trade_count == 1
    assert result.trace.expected_execution_cost == pytest.approx(0.003)


def test_utility_execution_policy_rejects_negative_utility_trade():
    from alpha_os.portfolio_execution_policy import (
        ExecutionPolicySpec,
        TradeTransitionRequest,
        apply_execution_policy,
    )

    result = apply_execution_policy(
        TradeTransitionRequest(
            desired_targets={"A": _target("A", 0.50)},
            current_weights={"A": 0.0},
            capital_base=10.0,
            execution_policy=ExecutionPolicySpec(),
            signal_value_by_subject={"A": 0.0},
            per_turnover_cost=0.01,
        )
    )

    transition = result.trace.subjects[0]
    assert result.executed_targets["A"].target_weight == pytest.approx(0.0)
    assert transition.reason == "negative_utility"
    assert transition.expected_trade_cost == pytest.approx(0.05)
    assert result.trace.negative_utility_trade_count == 1
    assert result.trace.utility_rejected_turnover == pytest.approx(0.50)


def test_utility_execution_policy_benefit_scale_changes_trade_utility():
    from alpha_os.portfolio_execution_policy import (
        ExecutionPolicySpec,
        TradeTransitionRequest,
        apply_execution_policy,
    )

    base_result = apply_execution_policy(
        TradeTransitionRequest(
            desired_targets={"A": _target("A", 0.50)},
            current_weights={"A": 0.0},
            capital_base=10.0,
            execution_policy=ExecutionPolicySpec(benefit_scale=1.0),
            signal_value_by_subject={"A": 0.02},
            per_turnover_cost=0.01,
        )
    )
    scaled_result = apply_execution_policy(
        TradeTransitionRequest(
            desired_targets={"A": _target("A", 0.50)},
            current_weights={"A": 0.0},
            capital_base=10.0,
            execution_policy=ExecutionPolicySpec(benefit_scale=2.0),
            signal_value_by_subject={"A": 0.02},
            per_turnover_cost=0.01,
        )
    )

    base_transition = base_result.trace.subjects[0]
    scaled_transition = scaled_result.trace.subjects[0]
    assert scaled_transition.expected_trade_benefit == pytest.approx(
        base_transition.expected_trade_benefit * 2.0
    )
    assert scaled_transition.trade_utility > base_transition.trade_utility


def test_utility_execution_policy_min_trade_utility_rejects_low_utility_trade():
    from alpha_os.portfolio_execution_policy import (
        ExecutionPolicySpec,
        TradeTransitionRequest,
        apply_execution_policy,
    )

    result = apply_execution_policy(
        TradeTransitionRequest(
            desired_targets={"A": _target("A", 0.50)},
            current_weights={"A": 0.0},
            capital_base=10.0,
            execution_policy=ExecutionPolicySpec(min_trade_utility=0.20),
            signal_value_by_subject={"A": 0.02},
            per_turnover_cost=0.0,
        )
    )

    transition = result.trace.subjects[0]
    assert transition.trade_utility == pytest.approx(0.10)
    assert transition.reason == "negative_utility"
    assert result.executed_targets["A"].target_weight == pytest.approx(0.0)
    assert result.trace.utility_rejected_turnover == pytest.approx(0.50)


def test_utility_execution_policy_can_disable_partial_fills():
    from alpha_os.portfolio_execution_policy import (
        ExecutionPolicySpec,
        TradeTransitionRequest,
        apply_execution_policy,
    )

    result = apply_execution_policy(
        TradeTransitionRequest(
            desired_targets={
                "A": _target("A", 0.30),
                "B": _target("B", 0.30),
            },
            current_weights={"A": 0.0, "B": 0.0},
            capital_base=1.0,
            execution_policy=ExecutionPolicySpec(
                turnover_budget=0.45,
                partial_fill_enabled=False,
            ),
            signal_value_by_subject={"A": 1.0, "B": 1.0},
        )
    )

    assert result.trace.executed_turnover == pytest.approx(0.30)
    assert result.trace.partial_fill_count == 0
    assert result.executed_targets["A"].target_weight == pytest.approx(0.30)
    assert result.executed_targets["B"].target_weight == pytest.approx(0.0)
    assert result.trace.subjects[1].reason == "turnover_budget"
