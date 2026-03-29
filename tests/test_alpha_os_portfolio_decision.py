from __future__ import annotations

import pytest


def test_portfolio_state_exposure_properties():
    from alpha_os.portfolio_decision import PortfolioPositionState, PortfolioState

    state = PortfolioState(
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
        PortfolioDecisionInput,
        PortfolioScalarInput,
        PortfolioState,
        PredictiveSignalInput,
    )

    decision_input = PortfolioDecisionInput(
        as_of="2026-03-29T00:00:00+00:00",
        portfolio_state=PortfolioState(),
        predictive_signals=(
            PredictiveSignalInput(
                source_id="corr_weighted_mean",
                subject_id="BTC",
                target_id="residual_return_3d",
                value=0.12,
                confidence=0.7,
            ),
        ),
        risk_inputs=(
            PortfolioScalarInput(
                name="realized_vol_3d",
                subject_id="BTC",
                value=0.18,
            ),
        ),
        cost_inputs=(
            PortfolioScalarInput(
                name="turnover_penalty",
                subject_id=None,
                value=0.01,
            ),
        ),
        uncertainty_inputs=(
            PortfolioScalarInput(
                name="score_instability",
                subject_id="BTC",
                value=0.2,
            ),
        ),
        dependence_inputs=(
            PortfolioScalarInput(
                name="hidden_bet_overlap",
                subject_id="BTC",
                value=0.4,
            ),
        ),
    )

    assert len(decision_input.predictive_signals) == 1
    assert len(decision_input.risk_inputs) == 1
    assert len(decision_input.cost_inputs) == 1
    assert len(decision_input.uncertainty_inputs) == 1
    assert len(decision_input.dependence_inputs) == 1


def test_portfolio_decision_output_exposure_properties():
    from alpha_os.portfolio_decision import PortfolioDecisionOutput, PortfolioTarget

    decision_output = PortfolioDecisionOutput(
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
