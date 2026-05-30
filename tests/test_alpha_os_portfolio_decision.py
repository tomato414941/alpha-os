from __future__ import annotations

import pytest


def _decision_input(
    *,
    portfolio_id: str | None = None,
    as_of: str | None = None,
    portfolio_state=None,
    predictive_signals=(),
    risk_inputs=(),
    cost_inputs=(),
    uncertainty_inputs=(),
    model_uncertainty_inputs=(),
    structural_uncertainty_inputs=(),
    dependence_inputs=(),
    historical_return_inputs=(),
    subject_metadata_by_subject=None,
):
    from alpha_os.portfolio_decision import (
        ObservedPortfolioInputs,
        PortfolioDecisionAssumptions,
        PortfolioDecisionInput,
        PortfolioState,
    )

    return PortfolioDecisionInput(
        portfolio_id=portfolio_id,
        as_of=as_of,
        portfolio_state=portfolio_state or PortfolioState(),
        observed_inputs=ObservedPortfolioInputs(
            predictive_signals=tuple(predictive_signals),
            risk_inputs=tuple(risk_inputs),
            uncertainty_inputs=tuple(uncertainty_inputs),
            model_uncertainty_inputs=tuple(model_uncertainty_inputs),
            structural_uncertainty_inputs=tuple(structural_uncertainty_inputs),
            dependence_inputs=tuple(dependence_inputs),
            historical_return_inputs=tuple(historical_return_inputs),
        ),
        assumptions=PortfolioDecisionAssumptions(
            cost_inputs=tuple(cost_inputs),
        ),
        subject_metadata_by_subject=subject_metadata_by_subject or {},
    )


def test_portfolio_state_exposure_properties():
    from alpha_os.portfolio_decision import PortfolioPositionState, PortfolioState

    state = PortfolioState(
        portfolio_id="paper_core",
        as_of="2026-03-29T00:00:00+00:00",
        positions=(
            PortfolioPositionState(subject_id="BTC", weight=0.3),
            PortfolioPositionState(subject_id="ETH", weight=-0.1),
        ),
        capital_base=2.0,
        gross_limit=1.2,
        net_limit=0.6,
        rebalance_step=4,
    )

    assert state.gross_exposure == pytest.approx(0.4)
    assert state.net_exposure == pytest.approx(0.2)
    assert state.weights_by_subject == {"BTC": 0.3, "ETH": -0.1}
    assert state.capital_base == pytest.approx(2.0)
    assert state.gross_limit == pytest.approx(1.2)
    assert state.net_limit == pytest.approx(0.6)
    assert state.rebalance_step == 4
    assert state.holding_period_days == 0
    assert state.recent_turnover == pytest.approx(0.0)
    assert state.current_drawdown == pytest.approx(0.0)


def test_subject_set_exposes_subject_ids_assets_and_signals():
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
    )

    subject_set = SubjectSet(
        subject_set_id="core_crypto",
        observation_specs=(
            ObservationSpec(
                observation_spec_id="btc_close",
                observable_id="daily_close",
            ),
            ObservationSpec(
                observation_spec_id="eth_close",
                observable_id="daily_close",
            ),
        ),
        bindings=(
            SubjectObservationBinding(
                subject_id="BTC_spot",
                asset="BTC",
                observation_spec_id="btc_close",
            ),
            SubjectObservationBinding(
                subject_id="ETH_spot",
                asset="ETH",
                observation_spec_id="eth_close",
            ),
        ),
    )

    assert subject_set.subject_set_id == "core_crypto"
    assert subject_set.subject_ids == ("BTC_spot", "ETH_spot")
    assert subject_set.asset_by_subject == {
        "BTC_spot": "BTC",
        "ETH_spot": "ETH",
    }
    assert subject_set.subject_kind_by_subject == {
        "BTC_spot": "asset",
        "ETH_spot": "asset",
    }
    assert subject_set.observation_spec_id_by_subject == {
        "BTC_spot": "btc_close",
        "ETH_spot": "eth_close",
    }


def test_subject_set_exposes_instrument_metadata_by_subject():
    from alpha_os.portfolio_decision import (
        InstrumentSpec,
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
    )

    subject_set = SubjectSet(
        subject_set_id="macro_futures",
        instruments=(
            InstrumentSpec(
                instrument_id="es_front",
                instrument_type="future",
                asset="ES",
                venue="CME",
                quote_ccy="USD",
                contract_family="ES",
                asset_class="equity_index",
                region="us",
                liquidity_tier="tier1",
                cluster="eq_index_dm",
                roll_rule="volume_switch",
                multiplier=50.0,
            ),
        ),
        observation_specs=(
            ObservationSpec(
                observation_spec_id="es_close",
                observable_id="daily_close",
                provided_observable_ids=(
                    "front_price",
                    "next_price",
                    "basis",
                ),
            ),
        ),
        bindings=(
            SubjectObservationBinding(
                subject_id="ES_front",
                subject_kind="future",
                asset="ES",
                observation_spec_id="es_close",
                instrument_id="es_front",
            ),
        ),
    )

    assert subject_set.instrument_id_by_subject == {"ES_front": "es_front"}
    instrument = subject_set.instrument_for_subject("ES_front")
    assert instrument is not None
    assert instrument.instrument_type == "future"
    assert instrument.venue == "CME"
    assert subject_set.asset_class_by_subject == {"ES_front": "equity_index"}
    assert subject_set.region_by_subject == {"ES_front": "us"}
    assert subject_set.liquidity_tier_by_subject == {"ES_front": "tier1"}
    assert subject_set.cluster_by_subject == {"ES_front": "eq_index_dm"}
    assert subject_set.subjects_grouped_by_instrument_field("asset_class") == {
        "equity_index": ("ES_front",)
    }
    assert (
        subject_set.observation_spec_for_subject("ES_front").provided_observable_ids
        == ("front_price", "next_price", "basis")
    )


def test_subject_set_supports_multiple_subject_kinds_without_backend_names():
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
    )

    subject_set = SubjectSet(
        subject_set_id="macro_mix",
        observation_specs=(
            ObservationSpec(
                observation_spec_id="spy_close",
                observable_id="daily_close",
            ),
            ObservationSpec(
                observation_spec_id="vix_close",
                observable_id="daily_close",
            ),
        ),
        bindings=(
            SubjectObservationBinding(
                subject_id="SPY_spot",
                asset="SPY",
                observation_spec_id="spy_close",
                subject_kind="equity",
            ),
            SubjectObservationBinding(
                subject_id="VIX_index",
                asset="VIX",
                observation_spec_id="vix_close",
                subject_kind="index",
            ),
        ),
    )

    assert subject_set.subject_kind_by_subject == {
        "SPY_spot": "equity",
        "VIX_index": "index",
    }
    assert subject_set.observation_spec_id_by_subject == {
        "SPY_spot": "spy_close",
        "VIX_index": "vix_close",
    }


def test_portfolio_decision_input_can_hold_multiple_input_kinds():
    from alpha_os.portfolio_decision import (
        CostInput,
        DependenceInput,
        PortfolioState,
        PredictiveSignalInput,
        RiskInput,
        UncertaintyInput,
    )

    decision_input = _decision_input(
        portfolio_id="paper_core",
        as_of="2026-03-29T00:00:00+00:00",
        portfolio_state=PortfolioState(holding_period_days=3),
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
                name="market_impact",
                subject_id="BTC",
                value=12.0,
                basis="per_notional",
                unit="bps",
            ),
        ),
        uncertainty_inputs=(
            UncertaintyInput(
                subject_id="BTC",
                source_id="corr_weighted_mean",
                target_id="residual_return_3d",
                estimate_std=0.2,
                basis="per_signal",
                proxy_components={"sample_coverage": 0.2},
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
    assert decision_input.uncertainty_inputs[0].estimate_std == pytest.approx(0.2)
    assert decision_input.dependence_inputs[0].right_subject_id == "ETH"


def test_portfolio_decision_input_assumptions_override_observed_costs():
    from alpha_os.portfolio_decision import (
        CostInput,
        ModelUncertaintyInput,
        ObservedPortfolioInputs,
        PortfolioDecisionAssumptions,
        PortfolioDecisionInput,
        StructuralUncertaintyInput,
    )

    decision_input = PortfolioDecisionInput(
        observed_inputs=ObservedPortfolioInputs(
            cost_inputs=(
                CostInput(
                    name="market_impact",
                    subject_id="BTC",
                    value=12.0,
                    basis="per_notional",
                    unit="bps",
                ),
            ),
        ),
        assumptions=PortfolioDecisionAssumptions(
            cost_inputs=(
                CostInput(
                    name="market_impact",
                    subject_id="BTC",
                    value=1.0,
                    basis="per_notional",
                    unit="bps",
                ),
            ),
            model_uncertainty_inputs=(
                ModelUncertaintyInput(
                    subject_id="BTC",
                    source_id="corr_weighted_mean",
                    target_id="residual_return_3d",
                    model_error=0.4,
                    basis="per_model",
                    proxy_components={"ensemble_instability": 0.4},
                ),
            ),
            structural_uncertainty_inputs=(
                StructuralUncertaintyInput(
                    subject_id="BTC",
                    source_id="corr_weighted_mean",
                    target_id="residual_return_3d",
                    structural_error=0.6,
                    basis="per_regime",
                    proxy_components={"regime_shift": 0.6},
                ),
            ),
        ),
    )

    assert len(decision_input.cost_inputs) == 1
    assert decision_input.cost_inputs[0].value == 1.0
    assert len(decision_input.model_uncertainty_inputs) == 1
    assert decision_input.model_uncertainty_inputs[0].model_error == pytest.approx(0.4)
    assert len(decision_input.structural_uncertainty_inputs) == 1
    assert decision_input.structural_uncertainty_inputs[0].structural_error == pytest.approx(0.6)


def test_portfolio_decision_output_exposure_properties():
    from alpha_os.portfolio_decision import PortfolioDecisionOutput, PortfolioTarget

    decision_output = PortfolioDecisionOutput(
        portfolio_id="paper_core",
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

