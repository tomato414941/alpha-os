from __future__ import annotations


def test_portfolio_state_from_decision_details_restores_snapshot():
    from alpha_os.portfolio_decision_inputs import portfolio_state_from_decision_details

    state = portfolio_state_from_decision_details(
        {
            "portfolio_state": {
                "portfolio_id": "paper_core",
                "as_of": "2026-03-26T00:00:00+00:00",
                "positions": [
                    {"subject_id": "BTC_spot", "weight": 0.15},
                    {"subject_id": "ETH_spot", "weight": -0.05},
                ],
                "capital_base": 2.5,
                "gross_limit": 1.2,
                "net_limit": 0.4,
                "rebalance_step": 7,
                "holding_period_days": 3,
                "recent_turnover": 0.1,
                "current_drawdown": 0.2,
            }
        }
    )

    assert state is not None
    assert state.portfolio_id == "paper_core"
    assert state.weights_by_subject == {"BTC_spot": 0.15, "ETH_spot": -0.05}
    assert state.capital_base == 2.5
    assert state.gross_limit == 1.2
    assert state.net_limit == 0.4
    assert state.rebalance_step == 7
    assert state.holding_period_days == 3
    assert state.recent_turnover == 0.1
    assert state.current_drawdown == 0.2


def test_build_runtime_observed_dependence_inputs_uses_aligned_correlation():
    from alpha_os.portfolio_decision_inputs import build_runtime_observed_dependence_inputs

    dependence_inputs = build_runtime_observed_dependence_inputs(
        subject_ids=("BTC_spot", "ETH_spot"),
        observation_series_by_subject={
            "BTC_spot": {"2026-03-24": 0.1, "2026-03-25": 0.2, "2026-03-26": 0.05},
            "ETH_spot": {"2026-03-24": 0.2, "2026-03-25": 0.4, "2026-03-26": 0.1},
        },
    )

    assert len(dependence_inputs) == 1
    assert dependence_inputs[0].left_subject_id == "BTC_spot"
    assert dependence_inputs[0].right_subject_id == "ETH_spot"
    assert dependence_inputs[0].basis == "correlation"
    assert dependence_inputs[0].value == 1.0


def test_observed_cost_estimates_scale_from_volatility():
    from alpha_os.portfolio_decision_inputs import (
        volatility_scaled_market_impact_bps,
    )

    market_impact_bps = volatility_scaled_market_impact_bps(0.12)

    assert market_impact_bps == 12.0
