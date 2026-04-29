from __future__ import annotations


def test_build_strategy_adaptation_family_weights_blends_adaptation_and_baseline_weights():
    from alpha_os.strategy_adaptation_weighting import (
        build_strategy_adaptation_family_weights,
    )
    from alpha_os.strategy_adaptation import FamilyReputation, StrategyAdaptationState

    state = StrategyAdaptationState(
        strategy_id="strategy:test",
        signal_train_id="signal-train:test",
        signal_discovery_id="discovery",
        source_evaluation_report_id="report-1",
        source_screening_result_id="discovery:screen:1",
        signal_reputations=(),
        family_reputations=(
            FamilyReputation(
                family_id="weak_family",
                mean_edge_score=0.20,
                mean_confidence=0.8,
                mean_stability_score=1.0,
                subject_coverage=2,
                member_count=2,
                update_count=1,
                updated_at="2026-04-03T00:00:00+00:00",
            ),
            FamilyReputation(
                family_id="strong_family",
                mean_edge_score=0.50,
                mean_confidence=0.9,
                mean_stability_score=1.2,
                subject_coverage=2,
                member_count=2,
                update_count=3,
                updated_at="2026-04-03T00:00:00+00:00",
            ),
        ),
        created_at="2026-04-03T00:00:00+00:00",
    )

    weights = build_strategy_adaptation_family_weights(
        family_ids=("weak_family", "strong_family"),
        strategy_adaptation_state=state,
        blend=0.2,
    )

    assert round(weights["weak_family"].baseline_weight, 6) == 0.5
    assert round(weights["strong_family"].baseline_weight, 6) == 0.5
    assert (
        weights["strong_family"].adaptation_weight
        > weights["weak_family"].adaptation_weight
    )
    assert weights["strong_family"].blended_weight > 0.5
    assert weights["weak_family"].blended_weight == 0.5


def test_build_strategy_adaptation_signal_weights_stays_near_baseline_with_one_update():
    from alpha_os.strategy_adaptation_weighting import (
        build_strategy_adaptation_signal_weights,
    )
    from alpha_os.strategy_adaptation import StrategyAdaptationState, SignalReputation

    state = StrategyAdaptationState(
        strategy_id="strategy:test",
        signal_train_id="signal-train:test",
        signal_discovery_id="discovery",
        source_evaluation_report_id="report-1",
        source_screening_result_id="discovery:screen:1",
        signal_reputations=(
            SignalReputation(
                signal_id="weak_hyp",
                family_id="family_a",
                subject_id="AAPL_equity",
                target_id="residual_return_3d",
                edge_score=0.1,
                mmc=0.01,
                confidence=0.4,
                stability_score=1.0,
                contribution_score=0.05,
                orientation=1,
                sample_count=20,
                update_count=1,
                updated_at="2026-04-03T00:00:00+00:00",
            ),
            SignalReputation(
                signal_id="strong_hyp",
                family_id="family_b",
                subject_id="AAPL_equity",
                target_id="residual_return_3d",
                edge_score=0.4,
                mmc=0.08,
                confidence=0.9,
                stability_score=2.0,
                contribution_score=0.25,
                orientation=1,
                sample_count=20,
                update_count=1,
                updated_at="2026-04-03T00:00:00+00:00",
            ),
        ),
        family_reputations=(),
        created_at="2026-04-03T00:00:00+00:00",
    )

    weights = build_strategy_adaptation_signal_weights(
        signal_ids=("weak_hyp", "strong_hyp"),
        strategy_adaptation_state=state,
        blend=0.2,
    )

    assert (
        weights["weak_hyp"].adaptation_multiplier
        < weights["strong_hyp"].adaptation_multiplier
    )
    assert round(weights["weak_hyp"].blended_multiplier, 6) == 1.0
    assert round(weights["strong_hyp"].blended_multiplier, 6) == 1.0
