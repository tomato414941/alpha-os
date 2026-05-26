from __future__ import annotations


def test_build_runtime_observed_inputs_translates_runtime_artifacts():
    from alpha_os.portfolio_decision_inputs import build_runtime_observed_inputs
    from alpha_os.store import MetaPredictionMetricState, MetaPredictionState

    observed = build_runtime_observed_inputs(
        meta_prediction=MetaPredictionState(
            evaluation_id="BTC:residual_return_3d:2026-03-26",
            subject_id="BTC_spot",
            asset="BTC",
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            value=0.25,
            contributor_count=2,
            details_json='{"contributors":[{"signal_id":"a","prediction":0.3},{"signal_id":"b","prediction":-0.1}]}',
            created_at="2026-03-26T00:00:00+00:00",
            updated_at="2026-03-26T00:00:00+00:00",
        ),
        metric=MetaPredictionMetricState(
            aggregation_kind="corr_weighted_mean",
            subject_id="BTC_spot",
            asset="BTC",
            target_id="residual_return_3d",
            corr=0.4,
            sample_count=10,
            window_size=20,
            start_evaluation_id=None,
            end_evaluation_id=None,
            updated_at="2026-03-26T00:00:00+00:00",
        ),
        subject_id="BTC_spot",
        target_id="residual_return_3d",
        aggregation_kind="corr_weighted_mean",
        risk_window=20,
        realized_volatility=0.12,
    )

    assert observed.predictive_signals[0].subject_id == "BTC_spot"
    assert observed.predictive_signals[0].confidence == 0.4
    assert observed.risk_inputs[0].name == "realized_vol_20"
    assert observed.risk_inputs[0].value == 0.12
    assert {item.name for item in observed.cost_inputs} == {"market_impact"}
    assert len(observed.uncertainty_inputs) == 1
    assert observed.uncertainty_inputs[0].source_id == "corr_weighted_mean"
    assert observed.uncertainty_inputs[0].target_id == "residual_return_3d"
    assert observed.uncertainty_inputs[0].estimate_std > 0.0
    assert observed.uncertainty_inputs[0].proxy_components == {
        "sample_coverage": 0.5,
        "ensemble_disagreement": 0.5,
        "contributor_dispersion": 0.5,
        "contributor_concentration": 0.0,
    }
    assert len(observed.model_uncertainty_inputs) == 1
    assert observed.model_uncertainty_inputs[0].source_id == "corr_weighted_mean"
    assert observed.model_uncertainty_inputs[0].target_id == "residual_return_3d"
    assert observed.model_uncertainty_inputs[0].model_error > 0.0
    assert observed.model_uncertainty_inputs[0].proxy_components == {
        "model_prediction_dispersion": 0.5,
        "model_weight_concentration": 0.0,
        "specification_weight_concentration": 0.0,
        "top_model_share": 0.5,
    }


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


def test_contributor_uncertainty_estimates_capture_dispersion_and_concentration():
    from alpha_os.portfolio_decision_inputs import (
        contributor_concentration,
        contributor_dispersion,
        specification_concentration,
        top_model_share,
    )
    from alpha_os.store import MetaPredictionState

    meta_prediction = MetaPredictionState(
        evaluation_id="BTC:residual_return_3d:2026-03-26",
        subject_id="BTC_spot",
        asset="BTC",
        target_id="residual_return_3d",
        aggregation_kind="corr_weighted_mean",
        value=0.2,
        contributor_count=3,
        details_json=(
            '{"contributors":['
            '{"signal_id":"reversal_1d@BTC_spot","prediction":0.4,"weight":0.8},'
            '{"signal_id":"average_gap_3d@BTC_spot","prediction":0.1,"weight":0.1},'
            '{"signal_id":"momentum_3d@BTC_spot","prediction":-0.2,"weight":0.1}'
            "]}"
        ),
        created_at="2026-03-26T00:00:00+00:00",
        updated_at="2026-03-26T00:00:00+00:00",
    )

    assert 0.0 < contributor_dispersion(meta_prediction) < 1.0
    assert 0.0 < contributor_concentration(meta_prediction) < 1.0
    assert 0.0 < specification_concentration(meta_prediction) < 1.0
    assert 0.0 < top_model_share(meta_prediction) < 1.0


def test_observed_cost_estimates_scale_from_volatility():
    from alpha_os.portfolio_decision_inputs import (
        volatility_scaled_market_impact_bps,
    )

    market_impact_bps = volatility_scaled_market_impact_bps(0.12)

    assert market_impact_bps == 12.0
