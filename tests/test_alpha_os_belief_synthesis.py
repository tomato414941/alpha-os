from __future__ import annotations


def test_build_signal_contributions_orients_negative_corr_and_attaches_regime_tags():
    from alpha_os.belief_synthesis import build_signal_contributions
    from alpha_os.screening import ScreeningCandidateResult

    survivors = (
        ScreeningCandidateResult(
            signal_id="trend__daily_close__lookback_40@AAPL_equity",
            specification_signal_id="trend__daily_close__lookback_40",
            family_id="trend__daily_close",
            subject_id="AAPL_equity",
            target_id="residual_return_3d",
            kind="momentum",
            lookback=40,
            score=0.20,
            corr=-0.20,
            stability_score=0.20 * (20.0**0.5),
            sample_count=20,
            keep=True,
            family_rank=1,
            reasons=(),
        ),
        ScreeningCandidateResult(
            signal_id="relative_strength__cross_sectional_return_rank_20d__lookback_10@AAPL_equity",
            specification_signal_id="relative_strength__cross_sectional_return_rank_20d__lookback_10",
            family_id="relative_strength__cross_sectional_return_rank_20d",
            subject_id="AAPL_equity",
            target_id="residual_return_3d",
            kind="relative_strength_rank",
            lookback=10,
            score=0.35,
            corr=0.35,
            stability_score=0.35 * (20.0**0.5),
            sample_count=20,
            keep=True,
            family_rank=1,
            reasons=(),
        ),
    )

    signal_contributions = build_signal_contributions(
        survivors=survivors,
        prediction_values_by_signal_id={
            "trend__daily_close__lookback_40@AAPL_equity": 0.10,
            "relative_strength__cross_sectional_return_rank_20d__lookback_10@AAPL_equity": 0.25,
        },
    )

    assert len(signal_contributions) == 2
    trend_contribution = signal_contributions[0]
    relative_contribution = signal_contributions[1]
    assert round(trend_contribution.oriented_prediction, 6) == -0.1
    assert trend_contribution.regime_tags == ("trend",)
    assert relative_contribution.regime_tags == (
        "cross_sectional",
        "relative_strength",
    )


def test_synthesize_beliefs_combines_signal_contributions_into_regime_aware_component():
    from alpha_os.belief_synthesis import SignalContribution, synthesize_beliefs

    signal_contributions = (
        SignalContribution(
            subject_id="AAPL_equity",
            target_id="residual_return_3d",
            signal_id="trend@AAPL_equity",
            family_id="trend_family",
            prediction_value=0.12,
            oriented_prediction=0.12,
            confidence=0.60,
            marginal_signal_contribution=0.60,
            stability_score=2.0,
            sample_count=20,
            regime_tags=("trend",),
        ),
        SignalContribution(
            subject_id="AAPL_equity",
            target_id="residual_return_3d",
            signal_id="low_vol_trend@AAPL_equity",
            family_id="low_vol_trend_family",
            prediction_value=0.09,
            oriented_prediction=0.09,
            confidence=0.55,
            marginal_signal_contribution=0.55,
            stability_score=1.8,
            sample_count=20,
            regime_tags=("low_vol", "trend"),
        ),
        SignalContribution(
            subject_id="AAPL_equity",
            target_id="residual_return_3d",
            signal_id="reversal@AAPL_equity",
            family_id="reversal_family",
            prediction_value=-0.04,
            oriented_prediction=-0.04,
            confidence=0.25,
            marginal_signal_contribution=0.25,
            stability_score=1.0,
            sample_count=20,
            regime_tags=("mean_reversion",),
        ),
    )

    synthesis = synthesize_beliefs(
        signal_discovery_id="multi_signal_search",
        screening_result_id="multi_signal_search:screen",
        signal_contributions=signal_contributions,
        created_at="2026-04-03T00:00:00+00:00",
    )

    assert len(synthesis.components) == 1
    component = synthesis.components[0]
    assert component.subject_id == "AAPL_equity"
    assert component.family_count == 3
    assert component.cluster_count == 2
    assert component.regime_tags == ("low_vol", "mean_reversion", "trend")
    assert component.representative_signal_ids == (
        "trend@AAPL_equity",
        "reversal@AAPL_equity",
    )
    assert component.belief_value > 0.0
    assert round(component.belief_value, 6) == 0.079643
    assert round(component.confidence, 6) == 0.290614
    assert round(component.mean_marginal_signal_contribution, 6) == 0.413043


def test_synthesize_beliefs_blends_strategy_adaptation_weights_with_baseline_weights():
    from alpha_os.belief_synthesis import SignalContribution, synthesize_beliefs
    from alpha_os.strategy_adaptation import (
        FamilyReputation,
        SignalReputation,
        StrategyAdaptationState,
    )

    signal_contributions = (
        SignalContribution(
            subject_id="AAPL_equity",
            target_id="residual_return_3d",
            signal_id="trend_fast@AAPL_equity",
            family_id="trend_family",
            prediction_value=0.10,
            oriented_prediction=0.10,
            confidence=0.50,
            marginal_signal_contribution=0.50,
            stability_score=1.5,
            sample_count=20,
            regime_tags=("trend",),
        ),
        SignalContribution(
            subject_id="AAPL_equity",
            target_id="residual_return_3d",
            signal_id="trend_slow@AAPL_equity",
            family_id="trend_family",
            prediction_value=0.02,
            oriented_prediction=0.02,
            confidence=0.50,
            marginal_signal_contribution=0.50,
            stability_score=1.5,
            sample_count=20,
            regime_tags=("trend",),
        ),
    )

    baseline = synthesize_beliefs(
        signal_discovery_id="discovery",
        screening_result_id="discovery:screen",
        signal_contributions=signal_contributions,
        created_at="2026-04-03T00:00:00+00:00",
    )
    strategy_adaptation_state = StrategyAdaptationState(
        strategy_id="strategy:test",
        signal_discovery_id="discovery",
        source_evaluation_report_id="report-1",
        source_screening_result_id="discovery:screen",
        signal_reputations=(
            SignalReputation(
                signal_id="trend_fast@AAPL_equity",
                family_id="trend_family",
                subject_id="AAPL_equity",
                target_id="residual_return_3d",
                orientation=1,
                edge_score=0.60,
                mmc=0.12,
                contribution_score=0.36,
                confidence=0.90,
                stability_score=1.5,
                sample_count=20,
                update_count=2,
                updated_at="2026-04-03T00:00:00+00:00",
            ),
            SignalReputation(
                signal_id="trend_slow@AAPL_equity",
                family_id="trend_family",
                subject_id="AAPL_equity",
                target_id="residual_return_3d",
                orientation=1,
                edge_score=0.20,
                mmc=0.01,
                contribution_score=0.105,
                confidence=0.70,
                stability_score=1.2,
                sample_count=20,
                update_count=1,
                updated_at="2026-04-03T00:00:00+00:00",
            ),
        ),
        family_reputations=(
            FamilyReputation(
                family_id="trend_family",
                mean_edge_score=0.40,
                mean_confidence=0.80,
                mean_stability_score=1.35,
                subject_coverage=1,
                member_count=2,
                update_count=2,
                updated_at="2026-04-03T00:00:00+00:00",
            ),
        ),
        created_at="2026-04-03T00:00:00+00:00",
    )

    with_meta = synthesize_beliefs(
        signal_discovery_id="discovery",
        screening_result_id="discovery:screen",
        signal_contributions=signal_contributions,
        created_at="2026-04-03T00:00:00+00:00",
        strategy_adaptation_state=strategy_adaptation_state,
        adaptation_blend=0.2,
    )

    assert round(baseline.components[0].belief_value, 6) == 0.06
    assert with_meta.components[0].belief_value > baseline.components[0].belief_value
    assert round(with_meta.components[0].belief_value, 6) == 0.060625
