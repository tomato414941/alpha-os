from __future__ import annotations

from types import SimpleNamespace


def test_screen_signals_applies_cheap_stability_and_redundancy_stages():
    from alpha_os.screening import ScreeningPolicy, screen_signals

    signals = [
        SimpleNamespace(
            signal_id="hyp_keep",
            signal_spec_id="spec_keep",
            subject_id="BTC_spot",
            target_id="residual_return_3d",
            kind="reversal",
            lookback=1,
        ),
        SimpleNamespace(
            signal_id="hyp_family_cap",
            signal_spec_id="spec_cap",
            subject_id="BTC_spot",
            target_id="residual_return_3d",
            kind="reversal",
            lookback=3,
        ),
        SimpleNamespace(
            signal_id="hyp_stability",
            signal_spec_id="spec_stability",
            subject_id="BTC_spot",
            target_id="residual_return_3d",
            kind="momentum",
            lookback=5,
        ),
        SimpleNamespace(
            signal_id="hyp_cheap",
            signal_spec_id="spec_cheap",
            subject_id="BTC_spot",
            target_id="residual_return_3d",
            kind="average_gap",
            lookback=3,
        ),
    ]

    metrics_by_id = {
        "hyp_keep": SimpleNamespace(corr=0.20, sample_count=25),
        "hyp_family_cap": SimpleNamespace(corr=0.15, sample_count=25),
        "hyp_stability": SimpleNamespace(corr=0.05, sample_count=9),
        "hyp_cheap": SimpleNamespace(corr=0.10, sample_count=4),
    }

    result = screen_signals(
        signals=signals,
        metrics_by_id=metrics_by_id,
        signal_discovery_id="search_a",
        policy=ScreeningPolicy(
            min_sample_count=5,
            min_abs_corr=0.04,
            min_stability_score=0.2,
            max_family_survivors_per_subject=1,
        ),
        family_ids_by_signal_spec_id={
            "spec_keep": "reversal_family",
            "spec_cap": "reversal_family",
            "spec_stability": "momentum_family",
            "spec_cheap": "average_gap_family",
        },
        family_budgets_by_family_id={"reversal_family": 1},
        created_at="2026-03-27T00:00:00+00:00",
    )

    by_signal_id = {item.signal_id: item for item in result.candidates}

    assert by_signal_id["hyp_keep"].keep is True
    assert by_signal_id["hyp_keep"].reasons == ()
    assert by_signal_id["hyp_keep"].stability_score == 1.0

    assert by_signal_id["hyp_family_cap"].keep is False
    assert by_signal_id["hyp_family_cap"].reasons == ("redundancy_family_cap",)

    assert by_signal_id["hyp_stability"].keep is False
    assert by_signal_id["hyp_stability"].reasons == ("stability_weak_signal",)
    assert round(by_signal_id["hyp_stability"].stability_score, 6) == 0.15

    assert by_signal_id["hyp_cheap"].keep is False
    assert by_signal_id["hyp_cheap"].reasons == ("cheap_insufficient_samples",)


def test_screen_signals_reduces_family_budget_adaptively():
    from alpha_os.screening import ScreeningPolicy, screen_signals

    signals = [
        SimpleNamespace(
            signal_id="hyp_a",
            signal_spec_id="spec_a",
            subject_id="BTC_spot",
            target_id="residual_return_3d",
            kind="reversal",
            lookback=1,
        ),
        SimpleNamespace(
            signal_id="hyp_b",
            signal_spec_id="spec_b",
            subject_id="BTC_spot",
            target_id="residual_return_3d",
            kind="reversal",
            lookback=3,
        ),
        SimpleNamespace(
            signal_id="hyp_c",
            signal_spec_id="spec_c",
            subject_id="BTC_spot",
            target_id="residual_return_3d",
            kind="reversal",
            lookback=5,
        ),
    ]

    metrics_by_id = {
        "hyp_a": SimpleNamespace(corr=0.08, sample_count=9),
        "hyp_b": SimpleNamespace(corr=0.07, sample_count=9),
        "hyp_c": SimpleNamespace(corr=0.06, sample_count=9),
    }

    result = screen_signals(
        signals=signals,
        metrics_by_id=metrics_by_id,
        signal_discovery_id="search_a",
        policy=ScreeningPolicy(
            min_sample_count=5,
            min_abs_corr=0.04,
            min_stability_score=0.05,
            adaptive_family_budget=True,
            adaptive_budget_stability_scale=10.0,
            max_family_survivors_per_subject=3,
        ),
        family_ids_by_signal_spec_id={
            "spec_a": "reversal_family",
            "spec_b": "reversal_family",
            "spec_c": "reversal_family",
        },
        family_budgets_by_family_id={"reversal_family": 3},
        created_at="2026-03-27T00:00:00+00:00",
    )

    by_signal_id = {item.signal_id: item for item in result.candidates}

    assert by_signal_id["hyp_a"].keep is True
    assert by_signal_id["hyp_b"].keep is True
    assert by_signal_id["hyp_c"].keep is False
    assert by_signal_id["hyp_c"].reasons == ("redundancy_adaptive_cap",)
