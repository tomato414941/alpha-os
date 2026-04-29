from __future__ import annotations

import pytest


def test_rebalance_friction_policy_round_trips_utility_execution_controls():
    from alpha_os.evaluation_cost_config import EvaluationRebalanceFrictionPolicySpec

    policy = EvaluationRebalanceFrictionPolicySpec.from_document(
        {
            "execution_mode": "utility_priority",
            "turnover_friction": 0.001,
            "no_trade_band": 0.01,
            "execution_cost_aversion": 2.0,
            "turnover_budget": 0.05,
            "benefit_scale": 1.5,
            "min_trade_utility": 0.02,
            "uncertainty_aversion": 0.75,
            "risk_aversion": 0.25,
            "partial_fill_enabled": "false",
        }
    )

    assert policy.benefit_scale == pytest.approx(1.5)
    assert policy.min_trade_utility == pytest.approx(0.02)
    assert policy.uncertainty_aversion == pytest.approx(0.75)
    assert policy.risk_aversion == pytest.approx(0.25)
    assert policy.partial_fill_enabled is False
    assert policy.to_document()["partial_fill_enabled"] is False


def test_rebalance_friction_policy_rejects_invalid_utility_controls():
    from alpha_os.evaluation_cost_config import EvaluationRebalanceFrictionPolicySpec

    with pytest.raises(ValueError, match="benefit_scale must be >= 0"):
        EvaluationRebalanceFrictionPolicySpec(benefit_scale=-1.0)
    with pytest.raises(ValueError, match="partial_fill_enabled must be boolean"):
        EvaluationRebalanceFrictionPolicySpec.from_document(
            {"partial_fill_enabled": "sometimes"}
        )
