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


def test_rule_based_policy_applies_signal_risk_uncertainty_and_cost():
    from alpha_os.portfolio_decision import (
        CostInput,
        PortfolioState,
        PredictiveSignalInput,
        RiskInput,
        UncertaintyInput,
    )
    from alpha_os.portfolio_sizing_policy import apply_signal_weighted_sizing

    decision_input = _decision_input(
        as_of="2026-03-29T00:00:00+00:00",
        portfolio_state=PortfolioState(holding_period_days=3),
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
                name="market_impact",
                subject_id="BTC",
                value=1000.0,
                basis="per_notional",
                unit="bps",
            ),
        ),
        uncertainty_inputs=(
            UncertaintyInput(
                subject_id="BTC",
                source_id="corr_weighted_mean",
                target_id="residual_return_3d",
                estimate_std=0.25,
                basis="per_signal",
                proxy_components={"sample_coverage": 0.25},
            ),
        ),
    )

    decision_output = apply_signal_weighted_sizing(decision_input)

    assert len(decision_output.targets) == 1
    assert decision_output.targets[0].subject_id == "BTC"
    assert decision_output.targets[0].target_weight == pytest.approx(0.151515, rel=1e-5)
    assert decision_output.targets[0].position_delta == pytest.approx(0.151515, rel=1e-5)
    assert decision_output.targets[0].risk_scale == pytest.approx(0.666667, rel=1e-5)
    assert decision_output.targets[0].entry_allowed is True


def test_rule_based_policy_respects_no_trade_band_and_gross_cap():
    from alpha_os.portfolio_decision import (
        CostInput,
        DependenceInput,
        PortfolioState,
        PredictiveSignalInput,
        RiskInput,
    )
    from alpha_os.portfolio_sizing_policy import apply_signal_weighted_sizing

    decision_input = _decision_input(
        as_of="2026-03-29T00:00:00+00:00",
        portfolio_state=PortfolioState(holding_period_days=3),
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

    decision_output = apply_signal_weighted_sizing(decision_input)
    targets_by_subject = {
        target.subject_id: target for target in decision_output.targets
    }

    assert targets_by_subject["BTC"].target_weight == pytest.approx(0.2)
    assert targets_by_subject["ETH"].target_weight == pytest.approx(0.2)
    assert targets_by_subject["SOL"].target_weight == pytest.approx(0.0)


def test_rule_based_policy_uses_drawdown_and_recent_turnover_state():
    from alpha_os.portfolio_decision import (
        PortfolioPositionState,
        PortfolioState,
        PredictiveSignalInput,
    )
    from alpha_os.portfolio_sizing_policy import apply_signal_weighted_sizing

    baseline_output = apply_signal_weighted_sizing(
        _decision_input(
            portfolio_state=PortfolioState(
                portfolio_id="paper_core",
                holding_period_days=3,
                positions=(PortfolioPositionState(subject_id="BTC", weight=0.0),),
            ),
            predictive_signals=(
                PredictiveSignalInput(
                    source_id="corr_weighted_mean",
                    source_kind="meta_prediction",
                    subject_id="BTC",
                    target_id="residual_return_3d",
                    value=1.0,
                ),
            ),
        )
    )
    same_position_output = apply_signal_weighted_sizing(
        _decision_input(
            portfolio_state=PortfolioState(
                portfolio_id="paper_core",
                holding_period_days=3,
                positions=(PortfolioPositionState(subject_id="BTC", weight=0.2),),
            ),
            predictive_signals=(
                PredictiveSignalInput(
                    source_id="corr_weighted_mean",
                    source_kind="meta_prediction",
                    subject_id="BTC",
                    target_id="residual_return_3d",
                    value=1.0,
                ),
            ),
        )
    )
    stressed_output = apply_signal_weighted_sizing(
        _decision_input(
            portfolio_state=PortfolioState(
                portfolio_id="paper_core",
                holding_period_days=3,
                positions=(PortfolioPositionState(subject_id="BTC", weight=0.2),),
                recent_turnover=0.5,
                current_drawdown=0.25,
            ),
            predictive_signals=(
                PredictiveSignalInput(
                    source_id="corr_weighted_mean",
                    source_kind="meta_prediction",
                    subject_id="BTC",
                    target_id="residual_return_3d",
                    value=1.0,
                ),
            ),
        )
    )

    assert baseline_output.targets[0].target_weight == pytest.approx(1.0)
    assert stressed_output.targets[0].target_weight < baseline_output.targets[0].target_weight
    assert stressed_output.targets[0].position_delta < same_position_output.targets[0].position_delta


def test_rule_based_policy_penalizes_short_holding_period():
    from alpha_os.portfolio_decision import (
        PortfolioPositionState,
        PortfolioState,
        PredictiveSignalInput,
    )
    from alpha_os.portfolio_sizing_policy import apply_signal_weighted_sizing

    early_output = apply_signal_weighted_sizing(
        _decision_input(
            portfolio_state=PortfolioState(
                holding_period_days=0,
                positions=(PortfolioPositionState(subject_id="BTC", weight=0.2),),
            ),
            predictive_signals=(
                PredictiveSignalInput(
                    source_id="corr_weighted_mean",
                    source_kind="meta_prediction",
                    subject_id="BTC",
                    target_id="residual_return_3d",
                    value=0.3,
                ),
            ),
        )
    )
    mature_output = apply_signal_weighted_sizing(
        _decision_input(
            portfolio_state=PortfolioState(
                holding_period_days=3,
                positions=(PortfolioPositionState(subject_id="BTC", weight=0.2),),
            ),
            predictive_signals=(
                PredictiveSignalInput(
                    source_id="corr_weighted_mean",
                    source_kind="meta_prediction",
                    subject_id="BTC",
                    target_id="residual_return_3d",
                    value=0.3,
                ),
            ),
        )
    )

    assert early_output.targets[0].target_weight < mature_output.targets[0].target_weight


def test_optimizer_policy_respects_weight_and_gross_constraints():
    from alpha_os.portfolio_decision import (
        PortfolioState,
        PredictiveSignalInput,
        RiskInput,
    )
    from alpha_os.portfolio_sizing_policy import (
        ConstrainedOptimizerSizingPolicy,
        apply_constrained_optimizer_sizing,
    )

    decision_input = _decision_input(
        as_of="2026-03-29T00:00:00+00:00",
        portfolio_state=PortfolioState(),
        predictive_signals=(
            PredictiveSignalInput(
                source_id="corr_weighted_mean",
                source_kind="meta_prediction",
                subject_id="BTC",
                target_id="residual_return_3d",
                value=0.8,
            ),
            PredictiveSignalInput(
                source_id="corr_weighted_mean",
                source_kind="meta_prediction",
                subject_id="ETH",
                target_id="residual_return_3d",
                value=0.7,
            ),
        ),
        risk_inputs=(
            RiskInput(
                name="gross_exposure_cap",
                subject_id=None,
                value=0.5,
                unit="weight",
            ),
        ),
    )

    decision_output = apply_constrained_optimizer_sizing(
        decision_input,
        sizing_policy=ConstrainedOptimizerSizingPolicy(max_abs_weight=0.4),
    )

    assert len(decision_output.targets) == 2
    assert all(
        abs(target.target_weight) <= 0.4 + 1e-6 for target in decision_output.targets
    )
    assert decision_output.gross_target_exposure == pytest.approx(0.5, rel=1e-5)


def test_optimizer_policy_uses_state_limits_and_capital_base():
    from alpha_os.portfolio_decision import PortfolioState, PredictiveSignalInput
    from alpha_os.portfolio_sizing_policy import (
        ConstrainedOptimizerSizingPolicy,
        apply_constrained_optimizer_sizing,
    )

    decision_input = _decision_input(
        portfolio_state=PortfolioState(
            capital_base=2.0,
            gross_limit=0.4,
            net_limit=0.2,
        ),
        predictive_signals=(
            PredictiveSignalInput(
                source_id="corr_weighted_mean",
                source_kind="meta_prediction",
                subject_id="BTC",
                target_id="residual_return_3d",
                value=0.8,
            ),
            PredictiveSignalInput(
                source_id="corr_weighted_mean",
                source_kind="meta_prediction",
                subject_id="ETH",
                target_id="residual_return_3d",
                value=0.7,
            ),
        ),
    )

    decision_output = apply_constrained_optimizer_sizing(
        decision_input,
        sizing_policy=ConstrainedOptimizerSizingPolicy(max_abs_weight=0.4),
    )

    assert decision_output.gross_target_exposure <= 0.4 + 1e-6
    assert abs(decision_output.net_target_exposure) <= 0.2 + 1e-6
    assert all(target.target_notional is not None for target in decision_output.targets)


def test_optimizer_policy_penalizes_turnover_from_current_weights():
    from alpha_os.portfolio_decision import (
        CostInput,
        PortfolioPositionState,
        PortfolioState,
        PredictiveSignalInput,
    )
    from alpha_os.portfolio_sizing_policy import (
        ConstrainedOptimizerSizingPolicy,
        apply_constrained_optimizer_sizing,
    )
    from alpha_os.portfolio_rebalance_friction import (
        PortfolioRebalanceFrictionPolicy,
    )

    decision_input = _decision_input(
        as_of="2026-03-29T00:00:00+00:00",
        portfolio_state=PortfolioState(
            positions=(PortfolioPositionState(subject_id="BTC", weight=0.4),),
            holding_period_days=1,
            recent_turnover=0.2,
        ),
        predictive_signals=(
            PredictiveSignalInput(
                source_id="corr_weighted_mean",
                source_kind="meta_prediction",
                subject_id="BTC",
                target_id="residual_return_3d",
                value=0.1,
            ),
        ),
        cost_inputs=(
            CostInput(
                name="turnover_cost_rate",
                subject_id=None,
                value=0.5,
                basis="per_turnover",
                unit="weight",
            ),
        ),
    )

    decision_output = apply_constrained_optimizer_sizing(
        decision_input,
        sizing_policy=ConstrainedOptimizerSizingPolicy(
            max_abs_weight=1.0,
        ),
        rebalance_friction_policy=PortfolioRebalanceFrictionPolicy(
            turnover_cost_aversion=1.0,
            signal_horizon_shortfall_aversion=1.0,
            recent_turnover_aversion=1.0,
        ),
    )

    assert len(decision_output.targets) == 1
    assert decision_output.targets[0].target_weight > 0.0
    assert abs(decision_output.targets[0].target_weight - 0.4) < 0.05
    assert abs(decision_output.targets[0].position_delta) < 0.05


def test_optimizer_policy_penalizes_model_uncertainty():
    from alpha_os.portfolio_decision import (
        ModelUncertaintyInput,
        PortfolioState,
        PredictiveSignalInput,
    )
    from alpha_os.portfolio_sizing_policy import (
        ConstrainedOptimizerSizingPolicy,
        apply_constrained_optimizer_sizing,
    )

    decision_input = _decision_input(
        portfolio_state=PortfolioState(gross_limit=1.0),
        predictive_signals=(
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="BTC",
                target_id="residual_return_3d",
                value=0.3,
                confidence=0.8,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="ETH",
                target_id="residual_return_3d",
                value=0.3,
                confidence=0.8,
            ),
        ),
        model_uncertainty_inputs=(
            ModelUncertaintyInput(
                source_id="belief",
                subject_id="BTC",
                target_id="residual_return_3d",
                model_error=0.05,
            ),
            ModelUncertaintyInput(
                source_id="belief",
                subject_id="ETH",
                target_id="residual_return_3d",
                model_error=0.5,
            ),
        ),
    )

    decision_output = apply_constrained_optimizer_sizing(
        decision_input,
        sizing_policy=ConstrainedOptimizerSizingPolicy(max_abs_weight=1.0),
    )
    targets_by_subject = {
        target.subject_id: target for target in decision_output.targets
    }

    assert targets_by_subject["BTC"].target_weight > targets_by_subject["ETH"].target_weight


def test_rule_based_policy_penalizes_model_uncertainty():
    from alpha_os.portfolio_decision import (
        ModelUncertaintyInput,
        PortfolioState,
        PredictiveSignalInput,
    )
    from alpha_os.portfolio_sizing_policy import apply_signal_weighted_sizing

    decision_input = _decision_input(
        portfolio_state=PortfolioState(),
        predictive_signals=(
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="BTC",
                target_id="residual_return_3d",
                value=0.3,
                confidence=0.8,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="ETH",
                target_id="residual_return_3d",
                value=0.3,
                confidence=0.8,
            ),
        ),
        model_uncertainty_inputs=(
            ModelUncertaintyInput(
                source_id="belief",
                subject_id="BTC",
                target_id="residual_return_3d",
                model_error=0.05,
            ),
            ModelUncertaintyInput(
                source_id="belief",
                subject_id="ETH",
                target_id="residual_return_3d",
                model_error=0.5,
            ),
        ),
    )

    decision_output = apply_signal_weighted_sizing(decision_input)
    targets_by_subject = {
        target.subject_id: target for target in decision_output.targets
    }

    assert targets_by_subject["BTC"].target_weight > targets_by_subject["ETH"].target_weight


def test_build_sizing_request_exposes_standardized_sizing_inputs():
    from alpha_os.portfolio_decision import (
        CostInput,
        DependenceInput,
        HistoricalReturnInput,
        PortfolioPositionState,
        PortfolioState,
        PredictiveSignalInput,
        RiskInput,
        UncertaintyInput,
    )
    from alpha_os.portfolio_sizing_policy import build_sizing_request

    decision_input = _decision_input(
        portfolio_state=PortfolioState(
            portfolio_id="paper_core",
            positions=(
                PortfolioPositionState(subject_id="AAA", weight=0.2),
                PortfolioPositionState(subject_id="BBB", weight=0.0),
            ),
            capital_base=2.0,
            gross_limit=1.1,
            net_limit=0.5,
            holding_period_days=2,
            recent_turnover=0.3,
            current_drawdown=0.1,
        ),
        predictive_signals=(
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="AAA",
                target_id="residual_return_5d",
                value=0.5,
                confidence=0.8,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="BBB",
                target_id="residual_return_5d",
                value=0.1,
                confidence=0.4,
            ),
        ),
        risk_inputs=(
            RiskInput(
                name="realized_vol_20",
                subject_id="AAA",
                value=0.2,
                horizon_days=20,
                unit="vol",
            ),
            RiskInput(
                name="gross_exposure_cap",
                subject_id=None,
                value=0.9,
                unit="weight",
            ),
        ),
        cost_inputs=(
            CostInput(
                name="market_impact",
                subject_id="AAA",
                value=25.0,
                basis="per_notional",
                unit="bps",
            ),
            CostInput(
                name="no_trade_band",
                subject_id="AAA",
                value=0.03,
                basis="per_delta_weight",
                unit="weight",
            ),
            CostInput(
                name="turnover_cost_rate",
                subject_id=None,
                value=0.02,
                basis="per_turnover",
                unit="weight",
            ),
        ),
        uncertainty_inputs=(
            UncertaintyInput(
                subject_id="AAA",
                source_id="belief",
                target_id="residual_return_5d",
                estimate_std=0.12,
                basis="compressed_belief",
                proxy_components={},
            ),
        ),
        dependence_inputs=(
            DependenceInput(
                name="correlation",
                left_subject_id="AAA",
                right_subject_id="BBB",
                value=0.4,
                basis="corr",
            ),
        ),
        historical_return_inputs=(
            HistoricalReturnInput(
                subject_id="AAA",
                returns_by_date={
                    "2026-01-01": 0.01,
                    "2026-01-02": -0.02,
                },
            ),
            HistoricalReturnInput(
                subject_id="BBB",
                returns_by_date={
                    "2026-01-01": 0.03,
                    "2026-01-02": 0.01,
                },
            ),
        ),
    )

    request = build_sizing_request(decision_input)

    assert request.subject_ids == ("AAA", "BBB")
    assert request.signal_values == pytest.approx((0.5, 0.1))
    assert request.current_weights == pytest.approx((0.2, 0.0))
    assert request.historical_return_matrix == (
        (0.01, 0.03),
        (-0.02, 0.01),
    )
    assert request.uncertainty_std == pytest.approx((0.12, 0.0))
    assert request.risk_values == pytest.approx((0.2, 0.0))
    assert request.no_trade_bands == pytest.approx((0.03, 0.0))
    assert request.market_impact_levels[0] == pytest.approx(0.0025)
    assert request.transaction_cost_levels[0] == pytest.approx(0.0025)
    assert request.short_cost_levels == pytest.approx((0.0, 0.0))
    assert request.gross_exposure_cap == pytest.approx(0.9)
    assert request.net_exposure_cap == pytest.approx(0.5)
    assert request.capital_base == pytest.approx(2.0)
    assert request.signal_horizons == (5, 5)
    assert request.turnover_cost_rate == pytest.approx(0.02)


def test_signed_mean_variance_policy_returns_signed_forecast_weights():
    from alpha_os.portfolio_decision import (
        HistoricalReturnInput,
        PortfolioState,
        PredictiveSignalInput,
        RiskInput,
    )
    from alpha_os.portfolio_sizing_policy import (
        SignedMeanVarianceSizingPolicy,
        apply_signed_mean_variance_sizing,
    )

    decision_input = _decision_input(
        portfolio_state=PortfolioState(gross_limit=3.0),
        predictive_signals=(
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="AAA",
                target_id="residual_return_5d",
                value=0.04,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="BBB",
                target_id="residual_return_5d",
                value=0.01,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="CCC",
                target_id="residual_return_5d",
                value=-0.03,
            ),
        ),
        risk_inputs=(
            RiskInput(name="realized_vol", subject_id="AAA", value=0.1),
            RiskInput(name="realized_vol", subject_id="BBB", value=0.1),
            RiskInput(name="realized_vol", subject_id="CCC", value=0.1),
        ),
        historical_return_inputs=(
            HistoricalReturnInput(
                subject_id="AAA",
                returns_by_date={"2026-01-01": 0.01},
            ),
            HistoricalReturnInput(
                subject_id="BBB",
                returns_by_date={"2026-01-01": 0.01},
            ),
            HistoricalReturnInput(
                subject_id="CCC",
                returns_by_date={"2026-01-01": 0.01},
            ),
        ),
    )

    decision_output = apply_signed_mean_variance_sizing(
        decision_input,
        sizing_policy=SignedMeanVarianceSizingPolicy(
            forecast_scale=1.0,
            risk_aversion=1.0,
            turnover_aversion=0.0,
            cost_aversion=0.0,
            short_cost_aversion=0.0,
            min_history_steps=3,
        ),
    )
    targets_by_subject = {
        target.subject_id: target for target in decision_output.targets
    }

    assert targets_by_subject["AAA"].target_weight > 0.0
    assert targets_by_subject["BBB"].target_weight > 0.0
    assert targets_by_subject["CCC"].target_weight < 0.0
    assert abs(targets_by_subject["AAA"].target_weight) > abs(
        targets_by_subject["BBB"].target_weight
    )
    assert decision_output.gross_target_exposure <= 3.0 + 1e-6


def test_signed_mean_variance_policy_penalizes_turnover_inside_optimizer():
    from alpha_os.portfolio_decision import (
        PortfolioPositionState,
        PortfolioState,
        PredictiveSignalInput,
        RiskInput,
    )
    from alpha_os.portfolio_sizing_policy import (
        SignedMeanVarianceSizingPolicy,
        apply_signed_mean_variance_sizing,
    )

    decision_input = _decision_input(
        portfolio_state=PortfolioState(
            gross_limit=1.0,
            positions=(PortfolioPositionState(subject_id="AAA", weight=0.4),),
        ),
        predictive_signals=(
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="AAA",
                target_id="residual_return_5d",
                value=-0.03,
            ),
        ),
        risk_inputs=(RiskInput(name="realized_vol", subject_id="AAA", value=0.1),),
    )

    low_penalty = apply_signed_mean_variance_sizing(
        decision_input,
        sizing_policy=SignedMeanVarianceSizingPolicy(
            turnover_aversion=0.0,
            cost_aversion=0.0,
            short_cost_aversion=0.0,
        ),
    )
    high_penalty = apply_signed_mean_variance_sizing(
        decision_input,
        sizing_policy=SignedMeanVarianceSizingPolicy(
            turnover_aversion=100.0,
            cost_aversion=0.0,
            short_cost_aversion=0.0,
        ),
    )

    assert abs(high_penalty.targets[0].position_delta) < abs(
        low_penalty.targets[0].position_delta
    )


def test_signed_mean_variance_policy_penalizes_short_cost():
    from alpha_os.portfolio_decision import (
        CostInput,
        PortfolioState,
        PredictiveSignalInput,
        RiskInput,
    )
    from alpha_os.portfolio_sizing_policy import (
        SignedMeanVarianceSizingPolicy,
        apply_signed_mean_variance_sizing,
    )

    base_input = _decision_input(
        portfolio_state=PortfolioState(gross_limit=1.0),
        predictive_signals=(
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="AAA",
                target_id="residual_return_5d",
                value=-0.03,
            ),
        ),
        risk_inputs=(RiskInput(name="realized_vol", subject_id="AAA", value=0.1),),
    )
    costly_short_input = _decision_input(
        portfolio_state=PortfolioState(gross_limit=1.0),
        predictive_signals=base_input.predictive_signals,
        risk_inputs=base_input.risk_inputs,
        cost_inputs=(
            CostInput(
                name="borrow_fee_bps_per_step",
                subject_id="AAA",
                value=500.0,
                basis="per_short_notional_per_step",
                unit="bps",
            ),
        ),
    )

    baseline = apply_signed_mean_variance_sizing(
        base_input,
        sizing_policy=SignedMeanVarianceSizingPolicy(
            turnover_aversion=0.0,
            cost_aversion=0.0,
            short_cost_aversion=1.0,
        ),
    )
    costly = apply_signed_mean_variance_sizing(
        costly_short_input,
        sizing_policy=SignedMeanVarianceSizingPolicy(
            turnover_aversion=0.0,
            cost_aversion=0.0,
            short_cost_aversion=1.0,
        ),
    )

    assert abs(costly.targets[0].target_weight) < abs(
        baseline.targets[0].target_weight
    )


def test_portfolio_sizing_policy_resolves_allocators_by_strategy():
    from alpha_os.portfolio_sizing_policy import (
        ConstrainedOptimizerAllocator,
        ConstrainedOptimizerSizingPolicy,
        HistoricalModelAllocator,
        HistoricalModelSizingPolicy,
        SignalWeightedAllocator,
        SignedMeanVarianceAllocator,
        SignedMeanVarianceSizingPolicy,
        portfolio_allocator_for_policy,
    )

    assert isinstance(portfolio_allocator_for_policy(None), SignalWeightedAllocator)
    assert isinstance(
        portfolio_allocator_for_policy(SignedMeanVarianceSizingPolicy()),
        SignedMeanVarianceAllocator,
    )
    assert isinstance(
        portfolio_allocator_for_policy(HistoricalModelSizingPolicy()),
        HistoricalModelAllocator,
    )
    assert isinstance(
        portfolio_allocator_for_policy(ConstrainedOptimizerSizingPolicy()),
        ConstrainedOptimizerAllocator,
    )


def test_skfolio_policy_applies_signal_direction_to_history_weights():
    from alpha_os.portfolio_decision import HistoricalReturnInput, PredictiveSignalInput
    from alpha_os.portfolio_sizing_policy import (
        HistoricalModelSizingPolicy,
        apply_portfolio_sizing_policy,
    )

    decision_input = _decision_input(
        predictive_signals=(
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="AAA",
                target_id="residual_return_5d",
                value=0.4,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="BBB",
                target_id="residual_return_5d",
                value=0.2,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="CCC",
                target_id="residual_return_5d",
                value=-0.1,
            ),
        ),
        historical_return_inputs=(
            HistoricalReturnInput(
                subject_id="AAA",
                returns_by_date={
                    "2026-01-01": 0.01,
                    "2026-01-02": 0.02,
                    "2026-01-03": -0.01,
                    "2026-01-04": 0.01,
                },
            ),
            HistoricalReturnInput(
                subject_id="BBB",
                returns_by_date={
                    "2026-01-01": -0.01,
                    "2026-01-02": 0.01,
                    "2026-01-03": 0.02,
                    "2026-01-04": 0.01,
                },
            ),
            HistoricalReturnInput(
                subject_id="CCC",
                returns_by_date={
                    "2026-01-01": 0.03,
                    "2026-01-02": -0.02,
                    "2026-01-03": -0.01,
                    "2026-01-04": 0.00,
                },
            ),
        ),
    )

    decision_output = apply_portfolio_sizing_policy(
        decision_input,
        sizing_policy=HistoricalModelSizingPolicy(
            model_type="hierarchical_risk_parity",
            min_history_steps=3,
        ),
    )
    targets_by_subject = {
        target.subject_id: target for target in decision_output.targets
    }

    assert targets_by_subject["AAA"].target_weight > 0.0
    assert targets_by_subject["BBB"].target_weight > 0.0
    assert targets_by_subject["CCC"].target_weight < 0.0
    assert sum(abs(target.target_weight) for target in decision_output.targets) == pytest.approx(1.0)


def test_conviction_adjusted_hrp_reduces_crypto_short_during_upward_breakout():
    from alpha_os.portfolio_decision import HistoricalReturnInput, PredictiveSignalInput
    from alpha_os.portfolio_sizing_policy import (
        HistoricalModelSizingPolicy,
        apply_portfolio_sizing_policy,
    )

    def history(base_return: float, cycle: int) -> dict[str, float]:
        return {
            f"2026-01-{index:03d}": base_return
            + ((index % cycle) - ((cycle - 1) / 2.0)) * 0.00005
            for index in range(260)
        }

    decision_input = _decision_input(
        predictive_signals=(
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="BTCUSDT_perp",
                target_id="residual_return_5d",
                value=-0.3,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="ZN_future",
                target_id="residual_return_5d",
                value=-0.3,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="ES_future",
                target_id="residual_return_5d",
                value=0.3,
            ),
        ),
        historical_return_inputs=(
            HistoricalReturnInput(
                subject_id="BTCUSDT_perp",
                returns_by_date=history(0.005, 5),
            ),
            HistoricalReturnInput(
                subject_id="ZN_future",
                returns_by_date=history(-0.002, 7),
            ),
            HistoricalReturnInput(
                subject_id="ES_future",
                returns_by_date=history(0.001, 11),
            ),
        ),
        subject_metadata_by_subject={
            "BTCUSDT_perp": {"asset_class": "crypto", "cluster": "crypto_major"},
            "ZN_future": {"asset_class": "rates", "cluster": "rates_us"},
            "ES_future": {"asset_class": "equity_index", "cluster": "eq_us"},
        },
    )

    baseline_output = apply_portfolio_sizing_policy(
        decision_input,
        sizing_policy=HistoricalModelSizingPolicy(
            model_type="hierarchical_risk_parity",
            min_history_steps=20,
        ),
    )
    adjusted_output = apply_portfolio_sizing_policy(
        decision_input,
        sizing_policy=HistoricalModelSizingPolicy(
            model_type="conviction_adjusted_hierarchical_risk_parity",
            min_history_steps=20,
        ),
    )
    baseline = {target.subject_id: target.target_weight for target in baseline_output.targets}
    adjusted = {target.subject_id: target.target_weight for target in adjusted_output.targets}
    baseline_gross = sum(abs(value) for value in baseline.values())
    adjusted_gross = sum(abs(value) for value in adjusted.values())

    assert adjusted["BTCUSDT_perp"] < 0.0
    assert adjusted["ZN_future"] < 0.0
    assert adjusted["ES_future"] > 0.0
    assert abs(adjusted["BTCUSDT_perp"]) < abs(baseline["BTCUSDT_perp"])
    assert (
        abs(adjusted["ZN_future"]) / abs(baseline["ZN_future"])
        > abs(adjusted["BTCUSDT_perp"]) / abs(baseline["BTCUSDT_perp"])
    )
    assert abs(adjusted["ZN_future"]) > abs(adjusted["BTCUSDT_perp"])
    assert adjusted_gross <= baseline_gross
    assert sum(adjusted.values()) == pytest.approx(0.0)


def test_diversified_risk_budget_respects_concentration_intent():
    from alpha_os.portfolio_concentration import (
        portfolio_effective_n,
        top_n_gross_share,
    )
    from alpha_os.portfolio_decision import HistoricalReturnInput, PredictiveSignalInput
    from alpha_os.portfolio_sizing_policy import (
        HistoricalModelSizingPolicy,
        apply_portfolio_sizing_policy,
    )

    def history(scale: float, cycle: int, phase: int) -> dict[str, float]:
        return {
            f"2026-01-{index:03d}": (
                ((index + phase) % cycle) - ((cycle - 1) / 2.0)
            )
            * scale
            + ((index % 5) - 2) * scale * 0.03
            for index in range(260)
        }

    decision_input = _decision_input(
        predictive_signals=(
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="AAA",
                target_id="residual_return_5d",
                value=0.4,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="BBB",
                target_id="residual_return_5d",
                value=-0.3,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="CCC",
                target_id="residual_return_5d",
                value=0.2,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="DDD",
                target_id="residual_return_5d",
                value=-0.1,
            ),
        ),
        historical_return_inputs=(
            HistoricalReturnInput(
                subject_id="AAA",
                returns_by_date=history(0.0001, 7, 0),
            ),
            HistoricalReturnInput(
                subject_id="BBB",
                returns_by_date=history(0.0050, 11, 2),
            ),
            HistoricalReturnInput(
                subject_id="CCC",
                returns_by_date=history(0.0060, 13, 4),
            ),
            HistoricalReturnInput(
                subject_id="DDD",
                returns_by_date=history(0.0070, 17, 6),
            ),
        ),
    )

    decision_output = apply_portfolio_sizing_policy(
        decision_input,
        sizing_policy=HistoricalModelSizingPolicy(
            model_type="diversified_risk_budget",
            min_history_steps=20,
            effective_n_floor=3.0,
            top_gross_share_cap_n=3,
            top_gross_share_cap=0.85,
        ),
    )
    weights = {target.subject_id: target.target_weight for target in decision_output.targets}

    assert portfolio_effective_n(weights.values()) >= 3.0 - 1e-6
    assert top_n_gross_share(weights.values(), top_n=3) <= 0.85 + 1e-6
    assert weights["AAA"] > 0.0
    assert weights["BBB"] < 0.0
    assert weights["CCC"] > 0.0
    assert weights["DDD"] < 0.0
    assert sum(abs(value) for value in weights.values()) == pytest.approx(1.0)


def test_skfolio_policy_ignores_missing_history_for_eligible_subjects():
    from alpha_os.portfolio_decision import HistoricalReturnInput, PredictiveSignalInput
    from alpha_os.portfolio_sizing_policy import (
        HistoricalModelSizingPolicy,
        apply_portfolio_sizing_policy,
    )

    decision_input = _decision_input(
        predictive_signals=(
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="AAA",
                target_id="residual_return_5d",
                value=0.4,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="BBB",
                target_id="residual_return_5d",
                value=0.2,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="CCC",
                target_id="residual_return_5d",
                value=-0.1,
            ),
        ),
        historical_return_inputs=(
            HistoricalReturnInput(
                subject_id="AAA",
                returns_by_date={
                    "2026-01-01": 0.01,
                    "2026-01-02": 0.02,
                    "2026-01-03": -0.01,
                    "2026-01-04": 0.01,
                },
            ),
            HistoricalReturnInput(
                subject_id="BBB",
                returns_by_date={
                    "2026-01-01": -0.01,
                    "2026-01-02": 0.01,
                    "2026-01-03": 0.02,
                    "2026-01-04": 0.01,
                },
            ),
        ),
    )

    decision_output = apply_portfolio_sizing_policy(
        decision_input,
        sizing_policy=HistoricalModelSizingPolicy(
            model_type="minimum_variance",
            min_history_steps=3,
        ),
    )
    targets_by_subject = {
        target.subject_id: target for target in decision_output.targets
    }

    assert targets_by_subject["AAA"].target_weight > 0.0
    assert targets_by_subject["BBB"].target_weight > 0.0
    assert targets_by_subject["CCC"].target_weight == pytest.approx(0.0)


def test_skfolio_policy_returns_zero_weights_when_all_signals_are_zero():
    from alpha_os.portfolio_decision import HistoricalReturnInput, PredictiveSignalInput
    from alpha_os.portfolio_sizing_policy import (
        HistoricalModelSizingPolicy,
        apply_portfolio_sizing_policy,
    )

    decision_input = _decision_input(
        predictive_signals=(
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="AAA",
                target_id="residual_return_5d",
                value=0.0,
            ),
            PredictiveSignalInput(
                source_id="belief",
                source_kind="compressed_belief",
                subject_id="BBB",
                target_id="residual_return_5d",
                value=0.0,
            ),
        ),
        historical_return_inputs=(
            HistoricalReturnInput(
                subject_id="AAA",
                returns_by_date={
                    "2026-01-01": 0.01,
                    "2026-01-02": 0.02,
                    "2026-01-03": -0.01,
                },
            ),
            HistoricalReturnInput(
                subject_id="BBB",
                returns_by_date={
                    "2026-01-01": -0.01,
                    "2026-01-02": 0.01,
                    "2026-01-03": 0.02,
                },
            ),
        ),
    )

    decision_output = apply_portfolio_sizing_policy(
        decision_input,
        sizing_policy=HistoricalModelSizingPolicy(
            model_type="equal_weight",
            min_history_steps=3,
        ),
    )

    assert all(target.target_weight == pytest.approx(0.0) for target in decision_output.targets)
