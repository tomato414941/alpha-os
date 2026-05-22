from __future__ import annotations

import pytest


def test_rebalance_friction_policy_round_trips_rebalance_controls():
    from alpha_os.evaluation_cost_config import EvaluationRebalanceFrictionPolicySpec

    policy = EvaluationRebalanceFrictionPolicySpec.from_document(
        {
            "turnover_friction": 0.001,
            "no_trade_band": 0.01,
            "execution_cost_aversion": 2.0,
            "turnover_budget": 0.05,
        }
    )

    assert policy.turnover_friction == pytest.approx(0.001)
    assert policy.no_trade_band == pytest.approx(0.01)
    assert policy.execution_cost_aversion == pytest.approx(2.0)
    assert policy.turnover_budget == pytest.approx(0.05)
    assert policy.to_document() == {
        "turnover_friction": pytest.approx(0.001),
        "no_trade_band": pytest.approx(0.01),
        "execution_cost_aversion": pytest.approx(2.0),
        "turnover_budget": pytest.approx(0.05),
    }


def test_rebalance_friction_policy_rejects_invalid_rebalance_controls():
    from alpha_os.evaluation_cost_config import EvaluationRebalanceFrictionPolicySpec

    with pytest.raises(ValueError, match="turnover_friction must be >= 0"):
        EvaluationRebalanceFrictionPolicySpec(turnover_friction=-1.0)
    with pytest.raises(ValueError, match="turnover_budget must be >= 0"):
        EvaluationRebalanceFrictionPolicySpec(turnover_budget=-0.1)
