from __future__ import annotations

import pytest


def test_build_portfolio_decision_input_uses_latest_meta_prediction(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.meta_metrics_service import refresh_target_meta_prediction_metrics
    from alpha_os.portfolio_decision import PortfolioState
    from alpha_os.portfolio_decision_service import build_portfolio_decision_input
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("reversal_1d")
        store.register_signal("average_gap_3d")

        values = [
            ("2026-03-24", 0.2, 0.0, 0.1),
            ("2026-03-25", 0.3, 0.1, 0.2),
            ("2026-03-26", 0.1, 0.05, 0.05),
        ]
        for date, pred_a, pred_b, obs in values:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="reversal_1d",
                prediction_value=pred_a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="average_gap_3d",
                prediction_value=pred_b,
                observation_value=obs,
            )

        refresh_target_meta_predictions(store)
        refresh_target_meta_prediction_metrics(store)

        decision_input = build_portfolio_decision_input(
            store,
            portfolio_state=PortfolioState(),
        )

        assert decision_input is not None
        assert decision_input.portfolio_state.portfolio_id is None
        assert decision_input.predictive_signals[0].source_kind == "meta_prediction"
        assert decision_input.predictive_signals[0].subject_id == "BTC"
        assert decision_input.predictive_signals[0].target_id == "residual_return_3d"
        assert decision_input.predictive_signals[0].confidence is not None
        assert 0.0 <= decision_input.predictive_signals[0].confidence <= 1.0
        assert decision_input.risk_inputs[0].name == "realized_vol_20"
        assert {item.name for item in decision_input.cost_inputs} == {
            "market_impact",
            "no_trade_band",
        }
        assert len(decision_input.uncertainty_inputs) == 1
        assert decision_input.uncertainty_inputs[0].source_id == "corr_weighted_mean"
        assert decision_input.uncertainty_inputs[0].estimate_std > 0.0
        assert set(decision_input.uncertainty_inputs[0].proxy_components) == {
            "sample_coverage",
            "ensemble_disagreement",
            "contributor_dispersion",
            "contributor_concentration",
        }
        assert len(decision_input.model_uncertainty_inputs) == 1
        assert decision_input.model_uncertainty_inputs[0].source_id == "corr_weighted_mean"
        assert decision_input.model_uncertainty_inputs[0].model_error > 0.0
        assert set(decision_input.model_uncertainty_inputs[0].proxy_components) == {
            "model_prediction_dispersion",
            "model_weight_concentration",
            "specification_weight_concentration",
            "top_model_share",
        }
    finally:
        store.close()


def test_build_portfolio_decision_input_from_compressed_belief(tmp_path):
    from alpha_os.compression import CompressedBelief, CompressedBeliefComponent
    from alpha_os.signal_discovery import SignalDiscoverySpec
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        PortfolioState,
        SubjectObservationBinding,
        SubjectSet,
        UniversePolicySpec,
    )
    from alpha_os.portfolio_decision_service import (
        build_portfolio_decision_input_from_compressed_belief,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.upsert_subject_set(
            "core_crypto",
            definition=SubjectSet(
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
                universe_policy=UniversePolicySpec(
                    base_currency="USD",
                    trading_calendar="24x7",
                    benchmark_id="core_crypto",
                ),
            ),
        )
        store.upsert_signal_discovery_spec(
            "search_a",
            definition=SignalDiscoverySpec(
                signal_discovery_id="search_a",
                subject_set_id="core_crypto",
                signal_spec_ids=("reversal_1d",),
            ),
        )
        for date, btc_obs, eth_obs in (
            ("2026-03-24", 0.10, 0.08),
            ("2026-03-25", 0.30, 0.05),
            ("2026-03-26", -0.20, -0.06),
            ("2026-03-27", 0.15, 0.04),
        ):
            store.finalize_observation(
                evaluation_id=f"BTC_spot:residual_return_3d:{date}",
                subject_id="BTC_spot",
                asset="BTC",
                target_id="residual_return_3d",
                observation_value=btc_obs,
            )
            store.finalize_observation(
                evaluation_id=f"ETH_spot:residual_return_3d:{date}",
                subject_id="ETH_spot",
                asset="ETH",
                target_id="residual_return_3d",
                observation_value=eth_obs,
            )
        store.upsert_compressed_belief(
            belief=CompressedBelief(
                compressed_belief_id="search_a:screen_a:compressed",
                signal_discovery_id="search_a",
                screening_result_id="search_a:screen_a",
                created_at="2026-03-27T00:00:00+00:00",
                components=(
                    CompressedBeliefComponent(
                        subject_id="BTC_spot",
                        target_id="residual_return_3d",
                        belief_value=0.12,
                        confidence=0.3,
                        signal_contribution_count=2,
                        family_ids=("reversal_family",),
                        signal_ids=("reversal_1d@BTC_spot", "reversal_3d@BTC_spot"),
                        representative_signal_ids=("reversal_1d@BTC_spot",),
                        family_count=1,
                        cluster_count=1,
                        effective_belief_count=1.0,
                        diversity_score=1.0,
                    ),
                    CompressedBeliefComponent(
                        subject_id="ETH_spot",
                        target_id="residual_return_3d",
                        belief_value=0.08,
                        confidence=0.5,
                        signal_contribution_count=2,
                        family_ids=("reversal_family",),
                        signal_ids=("reversal_1d@ETH_spot", "reversal_3d@ETH_spot"),
                        representative_signal_ids=("reversal_1d@ETH_spot",),
                        family_count=1,
                        cluster_count=1,
                        effective_belief_count=1.0,
                        diversity_score=1.0,
                    ),
                ),
            )
        )

        decision_input = build_portfolio_decision_input_from_compressed_belief(
            store,
            compressed_belief_id="search_a:screen_a:compressed",
            portfolio_state=PortfolioState(portfolio_id="paper_core"),
            portfolio_id="paper_core",
        )

        assert decision_input.portfolio_id == "paper_core"
        assert decision_input.as_of == "2026-03-27T00:00:00+00:00"
        assert len(decision_input.predictive_signals) == 2
        assert decision_input.predictive_signals[0].source_kind == "compressed_belief"
        assert decision_input.predictive_signals[0].value == 0.12
        assert len(decision_input.uncertainty_inputs) == 2
        assert all(item.estimate_std > 0.0 for item in decision_input.uncertainty_inputs)
        assert len(decision_input.model_uncertainty_inputs) == 2
        assert decision_input.model_uncertainty_inputs[0].model_error == 0.725
        assert decision_input.model_uncertainty_inputs[0].proxy_components == {
            "signal_contribution_count_inverse": 0.5,
            "family_count_inverse": 1.0,
            "cluster_count_inverse": 1.0,
            "effective_belief_count_inverse": 1.0,
            "diversity_gap": 0.0,
        }
        assert len(decision_input.risk_inputs) == 2
        assert {item.name for item in decision_input.cost_inputs} == {
            "market_impact",
            "no_trade_band",
        }
        assert len(decision_input.dependence_inputs) == 1
    finally:
        store.close()


def test_apply_decision_output_constraints_can_enforce_long_only_top_k():
    from alpha_os.portfolio_construction_config import PortfolioConstructionSpec
    from alpha_os.portfolio_decision import (
        PortfolioDecisionOutput,
        PortfolioState,
        PortfolioTarget,
    )
    from alpha_os.portfolio_decision_service import apply_decision_output_constraints

    decision_output = PortfolioDecisionOutput(
        portfolio_id="paper",
        as_of="2026-03-27T00:00:00+00:00",
        targets=(
            PortfolioTarget(subject_id="A", target_weight=0.6, position_delta=0.0),
            PortfolioTarget(subject_id="B", target_weight=0.3, position_delta=0.0),
            PortfolioTarget(subject_id="C", target_weight=-0.2, position_delta=0.0),
        ),
    )
    constrained = apply_decision_output_constraints(
        decision_output,
        portfolio_state=PortfolioState(
            portfolio_id="paper",
            capital_base=1.0,
            rebalance_step=1,
        ),
        portfolio_construction=PortfolioConstructionSpec(
            long_only=True,
            top_k=2,
            gross_exposure_cap=1.0,
        ),
    )

    targets = {item.subject_id: item for item in constrained.targets}
    assert set(targets) == {"A", "B", "C"}
    assert targets["C"].target_weight == 0.0
    assert targets["A"].target_weight + targets["B"].target_weight <= 1.0
    assert targets["A"].target_weight + targets["B"].target_weight == pytest.approx(0.9)
    assert targets["A"].target_weight > targets["B"].target_weight > 0.0


def test_apply_decision_output_constraints_defaults_to_rank_tilt_overlay():
    from alpha_os.portfolio_construction_config import PortfolioConstructionSpec
    from alpha_os.portfolio_decision import (
        PortfolioDecisionOutput,
        PortfolioState,
        PortfolioTarget,
    )
    from alpha_os.portfolio_decision_service import apply_decision_output_constraints

    decision_output = PortfolioDecisionOutput(
        portfolio_id="paper",
        as_of="2026-03-27T00:00:00+00:00",
        targets=(
            PortfolioTarget(subject_id="A", target_weight=0.5, position_delta=0.0),
            PortfolioTarget(subject_id="B", target_weight=0.3, position_delta=0.0),
            PortfolioTarget(subject_id="C", target_weight=0.2, position_delta=0.0),
        ),
    )
    constrained = apply_decision_output_constraints(
        decision_output,
        portfolio_state=PortfolioState(
            portfolio_id="paper",
            capital_base=1.0,
            rebalance_step=1,
        ),
        portfolio_construction=PortfolioConstructionSpec(
            long_only=True,
            gross_exposure_cap=1.0,
        ),
    )

    targets = {item.subject_id: item for item in constrained.targets}
    assert targets["A"].target_weight > 0.5
    assert targets["B"].target_weight == pytest.approx(0.3)
    assert 0.0 < targets["C"].target_weight < 0.2
    assert targets["C"].entry_allowed is True
    assert sum(item.target_weight for item in targets.values()) <= 1.0


def test_apply_decision_output_constraints_top_k_filters_after_overlay():
    from alpha_os.portfolio_construction_config import PortfolioConstructionSpec
    from alpha_os.portfolio_decision import (
        PortfolioDecisionOutput,
        PortfolioState,
        PortfolioTarget,
    )
    from alpha_os.portfolio_decision_service import apply_decision_output_constraints

    decision_output = PortfolioDecisionOutput(
        portfolio_id="paper",
        as_of="2026-03-27T00:00:00+00:00",
        targets=(
            PortfolioTarget(subject_id="A", target_weight=0.5, position_delta=0.0),
            PortfolioTarget(subject_id="B", target_weight=0.3, position_delta=0.0),
            PortfolioTarget(subject_id="C", target_weight=0.2, position_delta=0.0),
        ),
    )
    constrained = apply_decision_output_constraints(
        decision_output,
        portfolio_state=PortfolioState(
            portfolio_id="paper",
            capital_base=1.0,
            rebalance_step=1,
        ),
        portfolio_construction=PortfolioConstructionSpec(
            long_only=True,
            top_k=2,
            gross_exposure_cap=1.0,
        ),
    )

    targets = {item.subject_id: item for item in constrained.targets}
    assert targets["A"].target_weight > 0.5
    assert targets["B"].target_weight == pytest.approx(0.3)
    assert targets["C"].target_weight == pytest.approx(0.0)
    assert targets["C"].entry_allowed is False


def test_apply_decision_output_constraints_can_hold_between_rebalances():
    from alpha_os.portfolio_construction_config import PortfolioConstructionSpec
    from alpha_os.portfolio_decision import (
        PortfolioDecisionOutput,
        PortfolioPositionState,
        PortfolioState,
        PortfolioTarget,
    )
    from alpha_os.portfolio_decision_service import apply_decision_output_constraints

    decision_output = PortfolioDecisionOutput(
        portfolio_id="paper",
        as_of="2026-03-27T00:00:00+00:00",
        targets=(
            PortfolioTarget(subject_id="A", target_weight=1.0, position_delta=0.0),
            PortfolioTarget(subject_id="B", target_weight=0.0, position_delta=0.0),
        ),
    )
    constrained = apply_decision_output_constraints(
        decision_output,
        portfolio_state=PortfolioState(
            portfolio_id="paper",
            capital_base=1.0,
            rebalance_step=2,
            positions=(
                PortfolioPositionState(subject_id="A", weight=0.4),
                PortfolioPositionState(subject_id="B", weight=0.6),
            ),
        ),
        portfolio_construction=PortfolioConstructionSpec(
            rebalance_interval_steps=5,
            long_only=True,
            top_k=1,
            gross_exposure_cap=1.0,
        ),
    )

    targets = {item.subject_id: item for item in constrained.targets}
    assert targets["A"].target_weight == 0.4
    assert targets["B"].target_weight == 0.6
    assert all(item.position_delta == 0.0 for item in constrained.targets)


def test_apply_decision_output_constraints_can_scale_to_target_vol():
    from alpha_os.portfolio_construction_config import PortfolioConstructionSpec
    from alpha_os.portfolio_decision import (
        PortfolioDecisionOutput,
        PortfolioState,
        PortfolioTarget,
    )
    from alpha_os.portfolio_decision_service import apply_decision_output_constraints

    decision_output = PortfolioDecisionOutput(
        portfolio_id="paper",
        as_of="2026-03-27T00:00:00+00:00",
        targets=(
            PortfolioTarget(subject_id="A", target_weight=0.5, position_delta=0.0),
            PortfolioTarget(subject_id="B", target_weight=0.5, position_delta=0.0),
        ),
    )
    constrained = apply_decision_output_constraints(
        decision_output,
        portfolio_state=PortfolioState(
            portfolio_id="paper",
            capital_base=1.0,
            rebalance_step=1,
        ),
        portfolio_construction=PortfolioConstructionSpec(
            long_only=True,
            target_vol=0.1,
        ),
        risk_by_subject={"A": 0.2, "B": 0.2},
    )

    targets = {item.subject_id: item for item in constrained.targets}
    assert targets["A"].target_weight == pytest.approx(0.3535533906)
    assert targets["B"].target_weight == pytest.approx(0.3535533906)


def test_build_portfolio_decision_input_merges_explicit_assumptions(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.meta_metrics_service import refresh_target_meta_prediction_metrics
    from alpha_os.portfolio_decision import CostInput, DependenceInput, PortfolioState
    from alpha_os.portfolio_decision_service import (
        PortfolioDecisionAssumptions,
        build_portfolio_decision_input,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("reversal_1d")
        store.register_signal("average_gap_3d")

        values = [
            ("2026-03-24", 0.2, 0.0, 0.1),
            ("2026-03-25", 0.3, 0.1, 0.2),
            ("2026-03-26", 0.1, 0.05, 0.05),
        ]
        for date, pred_a, pred_b, obs in values:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="reversal_1d",
                prediction_value=pred_a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="average_gap_3d",
                prediction_value=pred_b,
                observation_value=obs,
            )

        refresh_target_meta_predictions(store)
        refresh_target_meta_prediction_metrics(store)

        decision_input = build_portfolio_decision_input(
            store,
            portfolio_id="paper_core",
            subject_id="BTC_spot",
            portfolio_state=PortfolioState(portfolio_id="paper_core"),
            assumptions=PortfolioDecisionAssumptions(
                cost_inputs=(
                    CostInput(
                        name="turnover_friction",
                        subject_id=None,
                        value=0.01,
                        basis="per_turnover",
                        unit="weight",
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
            ),
        )

        assert decision_input is not None
        assert decision_input.portfolio_id == "paper_core"
        assert decision_input.predictive_signals[0].subject_id == "BTC_spot"
        assert decision_input.risk_inputs[0].subject_id == "BTC_spot"
        assert {item.name for item in decision_input.cost_inputs} == {
            "market_impact",
            "no_trade_band",
            "turnover_friction",
        }
        assert decision_input.dependence_inputs[0].right_subject_id == "ETH"
    finally:
        store.close()


def test_build_portfolio_decision_input_adds_observed_dependence_from_portfolio_state(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.meta_metrics_service import refresh_target_meta_prediction_metrics
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        PortfolioPositionState,
        PortfolioState,
        SubjectObservationBinding,
        SubjectSet,
        UniversePolicySpec,
    )
    from alpha_os.portfolio_decision_service import (
        RuntimeDecisionBuildConfig,
        build_portfolio_decision_input,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("reversal_1d")
        store.register_signal("average_gap_3d")

        values = [
            ("2026-03-24", 0.2, 0.0, 0.1, 0.2),
            ("2026-03-25", 0.3, 0.1, 0.2, 0.4),
            ("2026-03-26", 0.1, 0.05, 0.05, 0.1),
        ]
        for date, pred_a, pred_b, obs_btc, obs_eth in values:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="reversal_1d",
                prediction_value=pred_a,
                observation_value=obs_btc,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="average_gap_3d",
                prediction_value=pred_b,
                observation_value=obs_btc,
            )
            store.finalize_observation(
                evaluation_id=f"ETH:residual_return_3d:{date}",
                observation_value=obs_eth,
                asset="ETH",
                target_id="residual_return_3d",
            )

        refresh_target_meta_predictions(store)
        refresh_target_meta_prediction_metrics(store)

        decision_input = build_portfolio_decision_input(
            store,
            portfolio_id="paper_core",
            subject_id="BTC_spot",
            portfolio_state=PortfolioState(
                portfolio_id="paper_core",
                positions=(PortfolioPositionState(subject_id="ETH_spot", weight=0.2),),
            ),
            config=RuntimeDecisionBuildConfig(
                dependence_window=20,
                subject_set=SubjectSet(
                    subject_set_id="core_crypto",
                    observation_specs=(
                        ObservationSpec(
                            observation_spec_id="eth_close",
                            observable_id="daily_close",
                        ),
                    ),
                    bindings=(
                        SubjectObservationBinding(
                            subject_id="ETH_spot",
                            asset="ETH",
                            observation_spec_id="eth_close",
                        ),
                    ),
                    universe_policy=UniversePolicySpec(
                        base_currency="USD",
                        trading_calendar="24x7",
                        benchmark_id="core_crypto",
                    ),
                ),
            ),
        )

        assert decision_input is not None
        assert len(decision_input.dependence_inputs) == 1
        assert decision_input.dependence_inputs[0].left_subject_id == "BTC_spot"
        assert decision_input.dependence_inputs[0].right_subject_id == "ETH_spot"
        assert decision_input.dependence_inputs[0].value == 1.0
    finally:
        store.close()


def test_build_portfolio_decision_input_includes_multiple_subject_signals(tmp_path):
    from alpha_os.portfolio_decision import PortfolioState
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
        UniversePolicySpec,
    )
    from alpha_os.portfolio_decision_service import (
        RuntimeDecisionBuildConfig,
        build_portfolio_decision_input,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.upsert_meta_prediction(
            evaluation_id="BTC:residual_return_3d:2026-03-26",
            asset="BTC",
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            value=0.2,
            contributor_count=2,
            details_json='{"contributors":[{"prediction":0.2},{"prediction":0.1}]}',
        )
        store.upsert_meta_prediction_metric(
            aggregation_kind="corr_weighted_mean",
            asset="BTC",
            target_id="residual_return_3d",
            corr=0.3,
            sample_count=10,
            window_size=20,
            start_evaluation_id=None,
            end_evaluation_id=None,
        )
        store.upsert_meta_prediction(
            evaluation_id="ETH:residual_return_3d:2026-03-26",
            asset="ETH",
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            value=0.1,
            contributor_count=2,
            details_json='{"contributors":[{"prediction":0.1},{"prediction":0.05}]}',
        )
        store.upsert_meta_prediction_metric(
            aggregation_kind="corr_weighted_mean",
            asset="ETH",
            target_id="residual_return_3d",
            corr=0.2,
            sample_count=8,
            window_size=20,
            start_evaluation_id=None,
            end_evaluation_id=None,
        )
        for date, btc_obs, eth_obs in (
            ("2026-03-24", 0.10, 0.20),
            ("2026-03-25", 0.20, 0.40),
            ("2026-03-26", 0.05, 0.10),
        ):
            store.finalize_observation(
                evaluation_id=f"BTC:residual_return_3d:{date}",
                observation_value=btc_obs,
                asset="BTC",
                target_id="residual_return_3d",
            )
            store.finalize_observation(
                evaluation_id=f"ETH:residual_return_3d:{date}",
                observation_value=eth_obs,
                asset="ETH",
                target_id="residual_return_3d",
            )

        decision_input = build_portfolio_decision_input(
            store,
            subject_id="BTC_spot",
            portfolio_state=PortfolioState(),
            config=RuntimeDecisionBuildConfig(
                subject_set=SubjectSet(
                    subject_set_id="core_crypto",
                    observation_specs=(
                        ObservationSpec(
                            observation_spec_id="eth_close",
                            observable_id="daily_close",
                        ),
                    ),
                    bindings=(
                        SubjectObservationBinding(
                            subject_id="ETH_spot",
                            asset="ETH",
                            observation_spec_id="eth_close",
                        ),
                    ),
                    universe_policy=UniversePolicySpec(
                        base_currency="USD",
                        trading_calendar="24x7",
                        benchmark_id="core_crypto",
                    ),
                ),
            ),
        )

        assert decision_input is not None
        assert {signal.subject_id for signal in decision_input.predictive_signals} == {
            "BTC_spot",
            "ETH_spot",
        }
        assert {risk_input.subject_id for risk_input in decision_input.risk_inputs} == {
            "BTC_spot",
            "ETH_spot",
        }
        assert len(decision_input.cost_inputs) == 4
        assert len(decision_input.uncertainty_inputs) == 2
        assert len(decision_input.model_uncertainty_inputs) == 2
        assert len(decision_input.dependence_inputs) == 1
    finally:
        store.close()


def test_build_portfolio_decision_input_rejects_incomplete_universe_policy(tmp_path):
    from alpha_os.portfolio_decision import PortfolioState
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
    )
    from alpha_os.portfolio_decision_service import (
        RuntimeDecisionBuildConfig,
        build_portfolio_decision_input,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.upsert_meta_prediction(
            evaluation_id="BTC:residual_return_3d:2026-03-26",
            asset="BTC",
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            value=0.2,
            contributor_count=2,
            details_json='{"contributors":[{"prediction":0.2},{"prediction":0.1}]}',
        )
        store.upsert_meta_prediction_metric(
            aggregation_kind="corr_weighted_mean",
            asset="BTC",
            target_id="residual_return_3d",
            corr=0.3,
            sample_count=10,
            window_size=20,
            start_evaluation_id=None,
            end_evaluation_id=None,
        )

        with pytest.raises(ValueError) as excinfo:
            build_portfolio_decision_input(
                store,
                subject_id="BTC_spot",
                portfolio_state=PortfolioState(),
                config=RuntimeDecisionBuildConfig(
                    subject_set=SubjectSet(
                        subject_set_id="core_crypto",
                        observation_specs=(
                            ObservationSpec(
                                observation_spec_id="eth_close",
                                observable_id="daily_close",
                            ),
                        ),
                        bindings=(
                            SubjectObservationBinding(
                                subject_id="ETH_spot",
                                asset="ETH",
                                observation_spec_id="eth_close",
                            ),
                            SubjectObservationBinding(
                                subject_id="SOL_spot",
                                asset="SOL",
                                observation_spec_id="eth_close",
                            ),
                        ),
                    ),
                ),
            )

        assert (
            "subject set universe policy is incomplete for multi-subject validation/evaluation: "
            "core_crypto missing base_currency, trading_calendar, benchmark_id"
            in str(excinfo.value)
        )
    finally:
        store.close()


def test_build_portfolio_decision_input_from_compressed_belief_rejects_incomplete_universe_policy(
    tmp_path,
):
    from alpha_os.compression import CompressedBelief, CompressedBeliefComponent
    from alpha_os.signal_discovery import SignalDiscoverySpec
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        PortfolioState,
        SubjectObservationBinding,
        SubjectSet,
    )
    from alpha_os.portfolio_decision_service import (
        build_portfolio_decision_input_from_compressed_belief,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.upsert_subject_set(
            "core_crypto",
            definition=SubjectSet(
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
            ),
        )
        store.upsert_signal_discovery_spec(
            "search_a",
            definition=SignalDiscoverySpec(
                signal_discovery_id="search_a",
                subject_set_id="core_crypto",
                signal_spec_ids=("reversal_1d",),
            ),
        )
        store.upsert_compressed_belief(
            belief=CompressedBelief(
                compressed_belief_id="search_a:screen_a:compressed",
                signal_discovery_id="search_a",
                screening_result_id="search_a:screen_a",
                created_at="2026-03-27T00:00:00+00:00",
                components=(
                    CompressedBeliefComponent(
                        subject_id="BTC_spot",
                        target_id="residual_return_3d",
                        belief_value=0.12,
                        confidence=0.3,
                        signal_contribution_count=2,
                        family_ids=("reversal_family",),
                        signal_ids=("reversal_1d@BTC_spot",),
                        representative_signal_ids=("reversal_1d@BTC_spot",),
                        family_count=1,
                        cluster_count=1,
                        effective_belief_count=1.0,
                        diversity_score=1.0,
                    ),
                    CompressedBeliefComponent(
                        subject_id="ETH_spot",
                        target_id="residual_return_3d",
                        belief_value=0.08,
                        confidence=0.5,
                        signal_contribution_count=2,
                        family_ids=("reversal_family",),
                        signal_ids=("reversal_1d@ETH_spot",),
                        representative_signal_ids=("reversal_1d@ETH_spot",),
                        family_count=1,
                        cluster_count=1,
                        effective_belief_count=1.0,
                        diversity_score=1.0,
                    ),
                ),
            )
        )

        with pytest.raises(ValueError) as excinfo:
            build_portfolio_decision_input_from_compressed_belief(
                store,
                compressed_belief_id="search_a:screen_a:compressed",
                portfolio_state=PortfolioState(portfolio_id="paper_core"),
                portfolio_id="paper_core",
            )

        assert (
            "subject set universe policy is incomplete for multi-subject validation/evaluation: "
            "core_crypto missing base_currency, trading_calendar, benchmark_id"
            in str(excinfo.value)
        )
    finally:
        store.close()


def test_build_portfolio_decision_output_returns_policy_result(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.meta_metrics_service import refresh_target_meta_prediction_metrics
    from alpha_os.portfolio_decision import (
        CostInput,
        PortfolioPositionState,
        PortfolioState,
    )
    from alpha_os.portfolio_decision_service import (
        PortfolioDecisionAssumptions,
        build_portfolio_decision_output,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("reversal_1d")
        store.register_signal("average_gap_3d")

        values = [
            ("2026-03-24", 0.4, 0.0, 0.2),
            ("2026-03-25", 0.3, 0.1, 0.1),
            ("2026-03-26", 0.2, 0.0, 0.05),
        ]
        for date, pred_a, pred_b, obs in values:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="reversal_1d",
                prediction_value=pred_a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="average_gap_3d",
                prediction_value=pred_b,
                observation_value=obs,
            )

        refresh_target_meta_predictions(store)
        refresh_target_meta_prediction_metrics(store)

        decision_output = build_portfolio_decision_output(
            store,
            portfolio_id="paper_core",
            subject_id="BTC_spot",
            portfolio_state=PortfolioState(
                portfolio_id="paper_core",
                positions=(PortfolioPositionState(subject_id="BTC_spot", weight=0.05),)
            ),
            assumptions=PortfolioDecisionAssumptions(
                cost_inputs=(
                    CostInput(
                        name="market_impact",
                        subject_id="BTC",
                        value=1000.0,
                        basis="per_notional",
                        unit="bps",
                    ),
                ),
            ),
        )

        assert decision_output is not None
        assert decision_output.portfolio_id == "paper_core"
        assert len(decision_output.targets) == 1
        assert decision_output.targets[0].subject_id == "BTC_spot"
        assert 0.0 <= decision_output.targets[0].target_weight <= 0.05
        assert decision_output.targets[0].position_delta < 0.0
        assert decision_output.targets[0].entry_allowed is True
    finally:
        store.close()


def test_build_portfolio_decision_output_accepts_optimizer_policy(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.meta_metrics_service import refresh_target_meta_prediction_metrics
    from alpha_os.portfolio_decision import PortfolioPositionState, PortfolioState
    from alpha_os.portfolio_sizing_policy import ConstrainedOptimizerSizingPolicy
    from alpha_os.portfolio_decision_service import build_portfolio_decision_output
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("reversal_1d")
        store.register_signal("average_gap_3d")

        values = [
            ("2026-03-24", 0.4, 0.0, 0.2),
            ("2026-03-25", 0.3, 0.1, 0.1),
            ("2026-03-26", 0.2, 0.0, 0.05),
        ]
        for date, pred_a, pred_b, obs in values:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="reversal_1d",
                prediction_value=pred_a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="average_gap_3d",
                prediction_value=pred_b,
                observation_value=obs,
            )

        refresh_target_meta_predictions(store)
        refresh_target_meta_prediction_metrics(store)

        decision_output = build_portfolio_decision_output(
            store,
            portfolio_id="paper_core",
            subject_id="BTC_spot",
            portfolio_state=PortfolioState(
                portfolio_id="paper_core",
                positions=(PortfolioPositionState(subject_id="BTC_spot", weight=0.05),),
            ),
            sizing_policy=ConstrainedOptimizerSizingPolicy(max_abs_weight=0.25),
        )

        assert decision_output is not None
        assert decision_output.portfolio_id == "paper_core"
        assert len(decision_output.targets) == 1
        assert decision_output.targets[0].subject_id == "BTC_spot"
        assert abs(decision_output.targets[0].target_weight) <= 0.25 + 1e-6
    finally:
        store.close()


def test_build_portfolio_decision_input_returns_none_without_meta_prediction(tmp_path):
    from alpha_os.portfolio_decision import PortfolioState
    from alpha_os.portfolio_decision_service import build_portfolio_decision_input
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        assert build_portfolio_decision_input(
            store,
            portfolio_state=PortfolioState(),
        ) is None
    finally:
        store.close()


def test_build_runtime_portfolio_state_uses_latest_persisted_decisions(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.meta_metrics_service import refresh_target_meta_prediction_metrics
    from alpha_os.portfolio_decision import PortfolioPositionState, PortfolioState
    from alpha_os.portfolio_decision_service import (
        build_portfolio_decision_output,
        build_runtime_portfolio_state,
        persist_portfolio_decision_output,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("reversal_1d")
        store.register_signal("average_gap_3d")

        values = [
            ("2026-03-24", 0.4, 0.0, 0.2),
            ("2026-03-25", 0.3, 0.1, 0.1),
            ("2026-03-26", 0.2, 0.0, 0.05),
        ]
        for date, pred_a, pred_b, obs in values:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="reversal_1d",
                prediction_value=pred_a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="average_gap_3d",
                prediction_value=pred_b,
                observation_value=obs,
            )

        refresh_target_meta_predictions(store)
        refresh_target_meta_prediction_metrics(store)

        initial_state = build_runtime_portfolio_state(
            store,
            portfolio_id="paper_core",
            aggregation_kind="corr_weighted_mean",
        )
        assert initial_state.positions == ()

        decision_output = build_portfolio_decision_output(
            store,
            portfolio_id="paper_core",
            portfolio_state=PortfolioState(
                portfolio_id="paper_core",
                as_of="2026-03-23T00:00:00+00:00",
                capital_base=2.0,
                gross_limit=0.8,
                net_limit=0.3,
                rebalance_step=5,
                recent_turnover=0.12,
                current_drawdown=0.08,
                positions=(PortfolioPositionState(subject_id="BTC", weight=0.05),),
            ),
        )
        assert decision_output is not None
        persist_portfolio_decision_output(
            store,
            decision_output=decision_output,
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            portfolio_state=PortfolioState(
                portfolio_id="paper_core",
                as_of="2026-03-23T00:00:00+00:00",
                capital_base=2.0,
                gross_limit=0.8,
                net_limit=0.3,
                rebalance_step=5,
                recent_turnover=0.12,
                current_drawdown=0.08,
                positions=(PortfolioPositionState(subject_id="BTC", weight=0.05),),
            ),
        )

        restored_state = build_runtime_portfolio_state(
            store,
            portfolio_id="paper_core",
            aggregation_kind="corr_weighted_mean",
        )
        assert len(restored_state.positions) == 1
        assert restored_state.positions[0].subject_id == "BTC"
        assert restored_state.positions[0].weight == decision_output.targets[0].target_weight
        assert restored_state.capital_base == 2.0
        assert restored_state.gross_limit == 0.8
        assert restored_state.net_limit == 0.3
        assert restored_state.rebalance_step == 6
        assert restored_state.holding_period_days >= 1
        assert restored_state.recent_turnover == 0.12
        assert restored_state.current_drawdown == 0.08
    finally:
        store.close()


def test_persist_portfolio_decision_output_writes_runtime_artifact(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.meta_metrics_service import refresh_target_meta_prediction_metrics
    from alpha_os.portfolio_decision import (
        CostInput,
        PortfolioPositionState,
        PortfolioState,
    )
    from alpha_os.portfolio_decision_service import (
        PortfolioDecisionAssumptions,
        build_portfolio_decision_input,
        build_portfolio_decision_output,
        persist_portfolio_decision_output,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("reversal_1d")
        store.register_signal("average_gap_3d")

        values = [
            ("2026-03-24", 0.4, 0.0, 0.2),
            ("2026-03-25", 0.3, 0.1, 0.1),
            ("2026-03-26", 0.2, 0.0, 0.05),
        ]
        for date, pred_a, pred_b, obs in values:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="reversal_1d",
                prediction_value=pred_a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="average_gap_3d",
                prediction_value=pred_b,
                observation_value=obs,
            )

        refresh_target_meta_predictions(store)
        refresh_target_meta_prediction_metrics(store)

        assumptions = PortfolioDecisionAssumptions(
            cost_inputs=(
                CostInput(
                    name="market_impact",
                    subject_id="BTC",
                    value=1000.0,
                    basis="per_notional",
                    unit="bps",
                ),
            ),
        )
        decision_output = build_portfolio_decision_output(
            store,
            portfolio_id="paper_core",
            portfolio_state=PortfolioState(
                portfolio_id="paper_core",
                positions=(PortfolioPositionState(subject_id="BTC", weight=0.05),),
            ),
            assumptions=assumptions,
        )
        assert decision_output is not None
        assert len(decision_output.targets) == 1
        decision_input = build_portfolio_decision_input(
            store,
            portfolio_id="paper_core",
            portfolio_state=PortfolioState(
                portfolio_id="paper_core",
                positions=(PortfolioPositionState(subject_id="BTC", weight=0.05),),
            ),
            assumptions=assumptions,
        )
        assert decision_input is not None

        persist_portfolio_decision_output(
            store,
            decision_output=decision_output,
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            assumptions=assumptions,
            decision_input=decision_input,
            sizing_method="signal_weighted",
            sizing_engine="optimizer",
        )

        items = store.list_portfolio_decisions(portfolio_id="paper_core", limit=10)
        assert len(items) == 1
        assert items[0].portfolio_id == "paper_core"
        assert items[0].subject_id == "BTC"
        assert items[0].target_id == "residual_return_3d"
        assert items[0].aggregation_kind == "corr_weighted_mean"
        assert items[0].details is not None
        assert items[0].details["sizing_method"] == "signal_weighted"
        assert items[0].details["sizing_engine"] == "optimizer"
        assert items[0].details["assumptions"]["cost_inputs"][0]["name"] == "market_impact"
        assert items[0].details["observed_inputs"]["cost_inputs"][0]["name"] == "market_impact"
        assert len(items[0].details["observed_inputs"]["model_uncertainty_inputs"]) == 1
        assert items[0].details["observed_inputs"]["structural_uncertainty_inputs"] == []
        assert items[0].details["assumptions"]["model_uncertainty_inputs"] == []
        assert items[0].details["assumptions"]["structural_uncertainty_inputs"] == []
        assert items[0].details["input_summary"]["subjects"]["BTC"]["predictive_signal"]["value"] is not None
        assert items[0].details["input_summary"]["subjects"]["BTC"]["model_uncertainty_inputs"][
            "corr_weighted_mean"
        ] > 0.0
        assert set(
            items[0].details["input_summary"]["subjects"]["BTC"]["model_uncertainty_proxies"]
        ) == {
            "model_prediction_dispersion",
            "model_weight_concentration",
            "specification_weight_concentration",
            "top_model_share",
        }
        assert items[0].details["input_summary"]["subjects"]["BTC"]["structural_uncertainty_inputs"] == {}
    finally:
        store.close()
