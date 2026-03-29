from __future__ import annotations

import pytest


def test_portfolio_state_exposure_properties():
    from alpha_os.portfolio_decision import PortfolioPositionState, PortfolioState

    state = PortfolioState(
        portfolio_id="paper_core",
        asset="BTC",
        as_of="2026-03-29T00:00:00+00:00",
        positions=(
            PortfolioPositionState(subject_id="BTC", weight=0.3),
            PortfolioPositionState(subject_id="ETH", weight=-0.1),
        ),
    )

    assert state.gross_exposure == pytest.approx(0.4)
    assert state.net_exposure == pytest.approx(0.2)
    assert state.weights_by_subject == {"BTC": 0.3, "ETH": -0.1}


def test_portfolio_decision_input_can_hold_multiple_input_kinds():
    from alpha_os.portfolio_decision import (
        CostInput,
        DependenceInput,
        PortfolioDecisionInput,
        PortfolioState,
        PredictiveSignalInput,
        RiskInput,
        UncertaintyInput,
    )

    decision_input = PortfolioDecisionInput(
        portfolio_id="paper_core",
        asset="BTC",
        as_of="2026-03-29T00:00:00+00:00",
        portfolio_state=PortfolioState(),
        predictive_signals=(
            PredictiveSignalInput(
                source_id="corr_weighted_mean",
                subject_id="BTC",
                target_id="residual_return_3d",
                value=0.12,
                confidence=0.7,
                source_kind="meta_prediction",
            ),
        ),
        risk_inputs=(
            RiskInput(
                name="realized_vol_3d",
                subject_id="BTC",
                value=0.18,
                horizon_days=3,
                unit="vol",
            ),
        ),
        cost_inputs=(
            CostInput(
                name="expected_slippage",
                subject_id="BTC",
                value=12.0,
                basis="per_notional",
                unit="bps",
            ),
        ),
        uncertainty_inputs=(
            UncertaintyInput(
                name="score_instability",
                subject_id="BTC",
                value=0.2,
                source_id="corr_weighted_mean",
                basis="per_signal",
            ),
        ),
        dependence_inputs=(
            DependenceInput(
                name="hidden_bet_overlap",
                left_subject_id="BTC",
                right_subject_id="ETH",
                value=0.4,
                basis="overlap",
            ),
        ),
    )

    assert len(decision_input.predictive_signals) == 1
    assert decision_input.portfolio_id == "paper_core"
    assert decision_input.predictive_signals[0].source_kind == "meta_prediction"
    assert len(decision_input.risk_inputs) == 1
    assert len(decision_input.cost_inputs) == 1
    assert decision_input.risk_inputs[0].horizon_days == 3
    assert decision_input.cost_inputs[0].basis == "per_notional"
    assert len(decision_input.uncertainty_inputs) == 1
    assert len(decision_input.dependence_inputs) == 1
    assert decision_input.uncertainty_inputs[0].source_id == "corr_weighted_mean"
    assert decision_input.dependence_inputs[0].right_subject_id == "ETH"


def test_portfolio_decision_output_exposure_properties():
    from alpha_os.portfolio_decision import PortfolioDecisionOutput, PortfolioTarget

    decision_output = PortfolioDecisionOutput(
        portfolio_id="paper_core",
        asset="BTC",
        as_of="2026-03-29T00:00:00+00:00",
        targets=(
            PortfolioTarget(
                subject_id="BTC",
                target_weight=0.25,
                position_delta=0.1,
                entry_allowed=True,
                risk_scale=0.8,
            ),
            PortfolioTarget(
                subject_id="ETH",
                target_weight=-0.05,
                position_delta=-0.05,
                entry_allowed=False,
                risk_scale=0.5,
            ),
        ),
    )

    assert decision_output.gross_target_exposure == pytest.approx(0.3)
    assert decision_output.net_target_exposure == pytest.approx(0.2)
    assert decision_output.portfolio_id == "paper_core"


def test_rule_based_policy_applies_signal_risk_uncertainty_and_cost():
    from alpha_os.portfolio_decision import (
        CostInput,
        PortfolioDecisionInput,
        PortfolioState,
        PredictiveSignalInput,
        RiskInput,
        UncertaintyInput,
    )
    from alpha_os.portfolio_decision_policy import apply_rule_based_policy

    decision_input = PortfolioDecisionInput(
        as_of="2026-03-29T00:00:00+00:00",
        portfolio_state=PortfolioState(),
        predictive_signals=(
            PredictiveSignalInput(
                source_id="corr_weighted_mean",
                source_kind="meta_prediction",
                subject_id="BTC",
                target_id="residual_return_3d",
                value=0.5,
                confidence=1.0,
            ),
        ),
        risk_inputs=(
            RiskInput(
                name="realized_vol_20d",
                subject_id="BTC",
                value=0.5,
                horizon_days=20,
                unit="vol",
            ),
        ),
        cost_inputs=(
            CostInput(
                name="expected_slippage",
                subject_id="BTC",
                value=1000.0,
                basis="per_notional",
                unit="bps",
            ),
        ),
        uncertainty_inputs=(
            UncertaintyInput(
                name="score_instability",
                subject_id="BTC",
                value=0.25,
                source_id="corr_weighted_mean",
                basis="per_signal",
            ),
        ),
    )

    decision_output = apply_rule_based_policy(decision_input)

    assert len(decision_output.targets) == 1
    assert decision_output.targets[0].subject_id == "BTC"
    assert decision_output.targets[0].target_weight == pytest.approx(0.222222, rel=1e-5)
    assert decision_output.targets[0].position_delta == pytest.approx(0.222222, rel=1e-5)
    assert decision_output.targets[0].risk_scale == pytest.approx(0.533333, rel=1e-5)
    assert decision_output.targets[0].entry_allowed is True


def test_rule_based_policy_respects_no_trade_band_and_gross_cap():
    from alpha_os.portfolio_decision import (
        CostInput,
        DependenceInput,
        PortfolioDecisionInput,
        PortfolioState,
        PredictiveSignalInput,
        RiskInput,
    )
    from alpha_os.portfolio_decision_policy import apply_rule_based_policy

    decision_input = PortfolioDecisionInput(
        as_of="2026-03-29T00:00:00+00:00",
        portfolio_state=PortfolioState(),
        predictive_signals=(
            PredictiveSignalInput(
                source_id="corr_weighted_mean",
                source_kind="meta_prediction",
                subject_id="BTC",
                target_id="residual_return_3d",
                value=0.6,
            ),
            PredictiveSignalInput(
                source_id="corr_weighted_mean",
                source_kind="meta_prediction",
                subject_id="ETH",
                target_id="residual_return_3d",
                value=0.6,
            ),
            PredictiveSignalInput(
                source_id="corr_weighted_mean",
                source_kind="meta_prediction",
                subject_id="SOL",
                target_id="residual_return_3d",
                value=0.02,
            ),
        ),
        risk_inputs=(
            RiskInput(
                name="gross_exposure_cap",
                subject_id=None,
                value=0.4,
                unit="weight",
            ),
        ),
        cost_inputs=(
            CostInput(
                name="no_trade_band",
                subject_id="SOL",
                value=0.05,
                basis="per_delta_weight",
                unit="weight",
            ),
        ),
        dependence_inputs=(
            DependenceInput(
                name="hidden_bet_overlap",
                left_subject_id="BTC",
                right_subject_id="ETH",
                value=1.0,
                basis="overlap",
            ),
        ),
    )

    decision_output = apply_rule_based_policy(decision_input)
    targets_by_subject = {
        target.subject_id: target for target in decision_output.targets
    }

    assert targets_by_subject["BTC"].target_weight == pytest.approx(0.2)
    assert targets_by_subject["ETH"].target_weight == pytest.approx(0.2)
    assert targets_by_subject["SOL"].target_weight == pytest.approx(0.0)
    assert targets_by_subject["SOL"].position_delta == pytest.approx(0.0)
    assert targets_by_subject["SOL"].entry_allowed is False
