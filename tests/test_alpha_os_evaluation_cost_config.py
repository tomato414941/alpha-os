from __future__ import annotations

import pytest


def test_rebalance_friction_policy_round_trips_rebalance_controls():
    from alpha_os.evaluation_cost_config import EvaluationRebalanceFrictionPolicySpec

    policy = EvaluationRebalanceFrictionPolicySpec.from_document(
        {
            "no_trade_band": 0.01,
            "turnover_budget": 0.05,
        }
    )

    assert policy.no_trade_band == pytest.approx(0.01)
    assert policy.turnover_budget == pytest.approx(0.05)
    assert policy.to_document() == {
        "no_trade_band": pytest.approx(0.01),
        "turnover_budget": pytest.approx(0.05),
    }


def test_rebalance_friction_policy_rejects_invalid_rebalance_controls():
    from alpha_os.evaluation_cost_config import EvaluationRebalanceFrictionPolicySpec

    with pytest.raises(ValueError, match="no_trade_band must be >= 0"):
        EvaluationRebalanceFrictionPolicySpec(no_trade_band=-1.0)
    with pytest.raises(ValueError, match="turnover_budget must be >= 0"):
        EvaluationRebalanceFrictionPolicySpec(turnover_budget=-0.1)


def test_trading_environment_round_trips_turnover_cost_rate():
    from alpha_os.evaluation_cost_config import TradingEnvironment

    environment = TradingEnvironment.from_document({"turnover_cost_rate": 0.001})

    assert environment.turnover_cost_rate == pytest.approx(0.001)
    assert environment.to_document()["turnover_cost_rate"] == pytest.approx(0.001)
