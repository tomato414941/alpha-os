from __future__ import annotations


def test_compress_screening_result_reduces_survivors_to_family_representatives():
    from alpha_os.compression import compress_screening_result
    from alpha_os.screening import ScreeningCandidateResult

    survivors = (
        ScreeningCandidateResult(
            signal_id="reversal_1d@BTC_spot",
            specification_signal_id="reversal_1d",
            family_id="reversal_family",
            subject_id="BTC_spot",
            target_id="residual_return_3d",
            kind="reversal",
            lookback=1,
            score=0.6,
            corr=0.6,
            stability_score=0.6 * (12.0**0.5),
            sample_count=12,
            keep=True,
            family_rank=1,
            reasons=(),
        ),
        ScreeningCandidateResult(
            signal_id="reversal_3d@BTC_spot",
            specification_signal_id="reversal_3d",
            family_id="reversal_family",
            subject_id="BTC_spot",
            target_id="residual_return_3d",
            kind="reversal",
            lookback=3,
            score=0.3,
            corr=0.3,
            stability_score=0.3 * (12.0**0.5),
            sample_count=12,
            keep=True,
            family_rank=2,
            reasons=(),
        ),
        ScreeningCandidateResult(
            signal_id="average_gap_3d@BTC_spot",
            specification_signal_id="average_gap_3d",
            family_id="average_gap_family",
            subject_id="BTC_spot",
            target_id="residual_return_3d",
            kind="average_gap",
            lookback=3,
            score=0.4,
            corr=0.4,
            stability_score=0.4 * (12.0**0.5),
            sample_count=12,
            keep=True,
            family_rank=1,
            reasons=(),
        ),
    )

    belief = compress_screening_result(
        signal_discovery_id="core_crypto_search",
        screening_result_id="core_crypto_search:screen",
        survivors=survivors,
        prediction_values_by_signal_id={
            "reversal_1d@BTC_spot": 0.10,
            "reversal_3d@BTC_spot": 0.20,
            "average_gap_3d@BTC_spot": -0.05,
        },
        created_at="2026-03-27T00:00:00+00:00",
    )

    assert len(belief.components) == 1
    component = belief.components[0]
    assert component.subject_id == "BTC_spot"
    assert component.family_ids == ("average_gap_family", "reversal_family")
    assert component.family_count == 2
    assert component.cluster_count == 2
    assert component.effective_belief_count >= 1.0
    assert component.representative_signal_ids == (
        "reversal_1d@BTC_spot",
        "average_gap_3d@BTC_spot",
    )
    assert component.signal_contribution_count == 3
    assert component.diversity_score > 0.9
    assert component.diversity_score <= 1.0
    assert round(component.belief_value, 6) == 0.047059
    assert round(component.confidence, 6) == 0.422069
    assert round(component.mean_marginal_signal_contribution, 6) == 0.34


def test_compress_screening_result_collapses_similar_families_into_single_cluster():
    from alpha_os.compression import compress_screening_result
    from alpha_os.screening import ScreeningCandidateResult

    survivors = (
        ScreeningCandidateResult(
            signal_id="momentum_1d@BTC_spot",
            specification_signal_id="momentum_1d",
            family_id="momentum_family",
            subject_id="BTC_spot",
            target_id="residual_return_3d",
            kind="momentum",
            lookback=1,
            score=0.7,
            corr=0.7,
            stability_score=0.7 * (12.0**0.5),
            sample_count=12,
            keep=True,
            family_rank=1,
            reasons=(),
        ),
        ScreeningCandidateResult(
            signal_id="range_position_5d@BTC_spot",
            specification_signal_id="range_position_5d",
            family_id="range_family",
            subject_id="BTC_spot",
            target_id="residual_return_3d",
            kind="range_position",
            lookback=5,
            score=0.65,
            corr=0.65,
            stability_score=0.65 * (12.0**0.5),
            sample_count=12,
            keep=True,
            family_rank=1,
            reasons=(),
        ),
        ScreeningCandidateResult(
            signal_id="average_gap_3d@BTC_spot",
            specification_signal_id="average_gap_3d",
            family_id="average_gap_family",
            subject_id="BTC_spot",
            target_id="residual_return_3d",
            kind="average_gap",
            lookback=3,
            score=0.2,
            corr=-0.2,
            stability_score=0.2 * (12.0**0.5),
            sample_count=12,
            keep=True,
            family_rank=1,
            reasons=(),
        ),
    )

    belief = compress_screening_result(
        signal_discovery_id="core_crypto_search",
        screening_result_id="core_crypto_search:screen",
        survivors=survivors,
        prediction_values_by_signal_id={
            "momentum_1d@BTC_spot": 0.10,
            "range_position_5d@BTC_spot": 0.11,
            "average_gap_3d@BTC_spot": -0.10,
        },
        created_at="2026-03-27T00:00:00+00:00",
    )

    component = belief.components[0]
    assert component.family_count == 3
    assert component.cluster_count == 1
    assert component.effective_belief_count == 1.0
    assert component.representative_signal_ids == ("momentum_1d@BTC_spot",)
    assert round(component.belief_value, 6) == 0.104194
    assert round(component.confidence, 6) == 1.0
    assert round(component.mean_marginal_signal_contribution, 6) == 0.491613


def test_compress_screening_result_flips_negative_corr_predictions_before_aggregation():
    from alpha_os.compression import compress_screening_result
    from alpha_os.screening import ScreeningCandidateResult

    survivors = (
        ScreeningCandidateResult(
            signal_id="momentum_1d@BTC_spot",
            specification_signal_id="momentum_1d",
            family_id="momentum_family",
            subject_id="BTC_spot",
            target_id="residual_return_3d",
            kind="momentum",
            lookback=1,
            score=0.7,
            corr=-0.7,
            stability_score=0.7 * (12.0**0.5),
            sample_count=12,
            keep=True,
            family_rank=1,
            reasons=(),
        ),
    )

    belief = compress_screening_result(
        signal_discovery_id="core_crypto_search",
        screening_result_id="core_crypto_search:screen",
        survivors=survivors,
        prediction_values_by_signal_id={
            "momentum_1d@BTC_spot": 0.10,
        },
        created_at="2026-03-27T00:00:00+00:00",
    )

    component = belief.components[0]
    assert round(component.belief_value, 6) == -0.1
