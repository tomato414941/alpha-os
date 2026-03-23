import sqlite3

import pytest

from alpha_os.hypotheses.quality import blend_quality
from alpha_os.hypotheses import (
    apply_allocation_rebalance_plan,
    backfill_observation_returns,
    build_allocation_rebalance_plan,
    HypothesisKind,
    HypothesisRecord,
    HypothesisStore,
    bootstrap_trust,
    compute_daily_contributions,
    is_research_backed,
    normalized_research_quality,
    record_daily_contributions,
    rolling_stake,
    target_stake,
    trust_score,
    is_capital_eligible,
    updated_stake,
    update_stakes_from_history,
    weighted_prediction,
)
from alpha_os.data.store import DataStore
from alpha_os.forward.tracker import ForwardTracker
from alpha_os.hypotheses.lifecycle import AllocationRebalanceEntry


def test_weighted_prediction_uses_stakes():
    value = weighted_prediction(
        {"h1": 1.0, "h2": -1.0},
        {"h1": 3.0, "h2": 1.0},
    )

    assert value == 0.5


def test_compute_daily_contributions_penalizes_redundant_hypotheses():
    contributions = compute_daily_contributions(
        {"h1": 1.0, "h2": 1.0},
        0.2,
        {"h1": 1.0, "h2": 1.0},
    )

    assert contributions["h1"] == 0.0
    assert contributions["h2"] == 0.0


def test_compute_daily_contributions_rewards_unique_hypothesis():
    contributions = compute_daily_contributions(
        {"h1": 1.0, "h2": -1.0},
        0.2,
        {"h1": 1.0, "h2": 1.0},
    )

    assert contributions["h1"] > 0
    assert contributions["h2"] < 0


def test_rolling_stake_uses_recent_mean():
    stake = rolling_stake(
        [0.5, 0.3, 0.1, -0.1, 0.2],
        lookback=3,
        min_observations=3,
        prior_stake=1.0,
    )

    assert stake == (0.5 + 0.3 + 0.1) / 3


def test_trust_score_clips_negative_values():
    assert trust_score(-1.0, -0.5) == 0.0


def test_normalized_research_quality_is_bounded():
    assert normalized_research_quality(3.0, metric="sharpe") == pytest.approx(1.0)
    assert normalized_research_quality(0.1, metric="log_growth") == pytest.approx(0.5)
    assert normalized_research_quality(-1.0, metric="sharpe") == pytest.approx(0.0)


def test_bootstrap_trust_scales_research_quality_conservatively():
    assert bootstrap_trust(2.0, metric="sharpe", bootstrap_weight=0.25) == pytest.approx(0.25)
    assert bootstrap_trust(0.10, metric="log_growth", bootstrap_weight=0.25) == pytest.approx(0.125)
    assert bootstrap_trust(
        2.0,
        metric="sharpe",
        bootstrap_weight=0.25,
        batch_research_weight=0.10,
        research_quality_source="batch_research_score",
    ) == pytest.approx(0.10)


def test_target_stake_applies_confidence_scaling():
    stake = target_stake(
        blended_quality=2.0,
        quality_confidence=0.25,
        marginal_contribution=0.4,
        quality_weight=1.0,
        marginal_contribution_weight=0.25,
    )

    assert stake == 0.25 * (2.0 + 0.25 * 0.4)


def test_target_stake_uses_bootstrap_trust_without_live_confidence():
    stake = target_stake(
        blended_quality=0.45,
        quality_confidence=0.0,
        marginal_contribution=0.0,
        research_quality=0.8,
        metric="log_growth",
        bootstrap_weight=0.25,
    )

    assert stake == pytest.approx(0.25 * min(0.8 / 0.20, 1.0))


def test_is_capital_eligible_requires_bootstrap_or_live_proven_thresholds():
    assert is_capital_eligible(
        research_quality=0.8,
        metric="sharpe",
        bootstrap_weight=0.25,
        has_min_observations=False,
    ) is True
    assert is_capital_eligible(
        research_quality=0.0,
        metric="sharpe",
        bootstrap_weight=0.25,
        has_min_observations=True,
        live_quality=0.10,
        marginal_contribution=0.01,
    ) is True
    assert is_capital_eligible(
        research_quality=0.0,
        metric="sharpe",
        bootstrap_weight=0.25,
        has_min_observations=False,
    ) is False
    assert is_capital_eligible(
        research_quality=0.8,
        research_quality_source="batch_research_score",
        metric="sharpe",
        bootstrap_weight=0.25,
        batch_research_weight=0.10,
        has_min_observations=False,
    ) is True


def test_is_research_backed_requires_quality_gate_for_batch_scores():
    assert is_research_backed(
        0.8,
        metric="sharpe",
        bootstrap_weight=0.25,
        batch_research_weight=0.10,
        batch_research_normalized_quality_min=0.10,
        research_quality_source="batch_research_score",
    ) is True
    assert is_research_backed(
        0.1,
        metric="sharpe",
        bootstrap_weight=0.25,
        batch_research_weight=0.10,
        batch_research_normalized_quality_min=0.10,
        research_quality_source="batch_research_score",
    ) is False


def test_updated_stake_smooths_toward_target():
    stake = updated_stake(
        current_stake=1.0,
        target_stake_value=0.5,
        stake_update_rate=0.10,
    )

    assert stake == pytest.approx(0.95)


def test_update_stakes_from_history_updates_store(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.DSL,
            definition={"expression": "f1"},
            stake=1.0,
            metadata={"oos_log_growth": 0.8},
        )
    )
    for idx, contribution in enumerate([0.5, 0.3, 0.1], start=1):
        store.record_contribution(
            "h1",
            date=f"2026-03-0{idx}",
            contribution=contribution,
        )

    updates = update_stakes_from_history(
        store,
        metric="log_growth",
        lookback=3,
        min_observations=3,
        full_weight_observations=3,
    )
    updated = store.get("h1")
    estimate = blend_quality(
        0.8,
        [0.1, 0.3, 0.5],
        metric="log_growth",
        rolling_window=3,
        min_observations=3,
        full_weight_observations=3,
    )
    expected = updated_stake(
        1.0,
        target_stake(
            estimate.blended_quality,
            estimate.confidence,
            (0.5 + 0.3 + 0.1) / 3,
        ),
    )

    assert updates["h1"] == expected
    assert updated is not None
    assert updated.stake == expected
    assert updated.metadata["lifecycle_blended_quality"] == pytest.approx(
        estimate.blended_quality
    )
    assert updated.metadata["lifecycle_quality_confidence"] == pytest.approx(
        estimate.confidence
    )
    assert updated.metadata["lifecycle_marginal_contribution"] == pytest.approx(
        (0.5 + 0.3 + 0.1) / 3
    )
    assert updated.metadata["lifecycle_bootstrap_trust"] == pytest.approx(0.25)
    assert updated.metadata["lifecycle_live_quality"] == pytest.approx(
        estimate.live_quality
    )
    assert updated.metadata["lifecycle_raw_live_quality"] == pytest.approx(
        estimate.raw_live_quality
    )
    assert updated.metadata["lifecycle_target_stake"] == pytest.approx(
        target_stake(
            estimate.blended_quality,
            estimate.confidence,
            (0.5 + 0.3 + 0.1) / 3,
            research_quality=0.8,
            metric="log_growth",
        )
    )
    store.close()


def test_update_stakes_from_history_uses_live_returns_for_quality(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.DSL,
            definition={"expression": "f1"},
            stake=1.0,
            metadata={"oos_log_growth": 0.0},
        )
    )
    store.record_contribution(
        "h1",
        date="2026-03-01",
        contribution=0.0,
    )

    updates = update_stakes_from_history(
        store,
        metric="log_growth",
        lookback=3,
        min_observations=1,
        full_weight_observations=1,
        live_returns_for=lambda _hypothesis_id: [0.2],
    )
    updated = store.get("h1")

    assert "h1" in updates
    assert updated is not None
    assert updated.metadata["lifecycle_quality_confidence"] == pytest.approx(1.0)
    assert updated.metadata["lifecycle_blended_quality"] > 0.0
    assert updated.metadata["lifecycle_live_quality"] > 0.0
    assert updated.metadata["lifecycle_raw_live_quality"] > 0.0
    assert updated.metadata["lifecycle_bootstrap_trust"] == pytest.approx(0.0)
    store.close()


def test_update_stakes_from_history_keeps_unscored_unproven_at_zero(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.DSL,
            definition={"expression": "f1"},
            stake=0.0,
            metadata={},
        )
    )

    updates = update_stakes_from_history(
        store,
        metric="sharpe",
        min_observations=5,
        full_weight_observations=63,
        live_returns_for=lambda _hypothesis_id: [0.001, 0.0012],
    )
    updated = store.get("h1")

    assert updates == {}
    assert updated is not None
    assert updated.stake == pytest.approx(0.0)
    assert updated.metadata["lifecycle_target_stake"] == pytest.approx(0.0)
    store.close()


def test_build_allocation_rebalance_plan_zeroes_unscored_unproven_hypotheses(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.DSL,
            definition={"expression": "f1"},
            stake=1.0,
            metadata={},
        )
    )

    plan = build_allocation_rebalance_plan(
        store,
        metric="sharpe",
        min_observations=5,
        full_weight_observations=63,
        live_returns_for=lambda _hypothesis_id: [0.01, -0.01],
    )

    assert len(plan) == 1
    entry = plan[0]
    assert entry.research_backed is False
    assert entry.research_retained is False
    assert entry.live_proven is False
    assert entry.capital_eligible is False
    assert entry.capital_reason == "none"
    assert entry.live_promotion_blocker == "insufficient_observations"
    assert entry.proposed_stake == pytest.approx(0.0)
    store.close()


def test_build_allocation_rebalance_plan_promotes_live_proven_unscored_hypothesis(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.DSL,
            definition={"expression": "f1"},
            stake=0.0,
            metadata={},
        )
    )
    for idx, contribution in enumerate([0.02, 0.03, 0.025], start=1):
        store.record_contribution(
            "h1",
            date=f"2026-03-0{idx}",
            contribution=contribution,
        )

    plan = build_allocation_rebalance_plan(
        store,
        metric="log_growth",
        min_observations=3,
        full_weight_observations=3,
        live_proven_quality_min=0.01,
        live_proven_marginal_contribution_min=0.0,
        live_returns_for=lambda _hypothesis_id: [0.02, 0.015, 0.01],
    )

    entry = plan[0]
    assert entry.research_backed is False
    assert entry.research_retained is False
    assert entry.live_proven is True
    assert entry.capital_eligible is True
    assert entry.capital_reason == "live_proven"
    assert entry.live_promotion_blocker == "eligible"
    assert entry.proposed_stake > 0.0
    store.close()


def test_build_allocation_rebalance_plan_demotes_weak_research_backed_hypothesis(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.ML,
            definition={"model_ref": "m1"},
            stake=1.0,
            metadata={"oos_sharpe": 0.8},
        )
    )
    for idx, contribution in enumerate([-0.02, -0.03, -0.01], start=1):
        store.record_contribution(
            "h1",
            date=f"2026-03-0{idx}",
            contribution=contribution,
        )

    plan = build_allocation_rebalance_plan(
        store,
        metric="sharpe",
        min_observations=3,
        full_weight_observations=3,
        bootstrap_retention_quality_min=0.0,
        bootstrap_retention_marginal_contribution_min=0.0,
        live_proven_quality_min=0.05,
        live_proven_marginal_contribution_min=0.0,
        live_returns_for=lambda _hypothesis_id: [-0.02, -0.015, -0.01],
    )

    entry = plan[0]
    assert entry.research_backed is True
    assert entry.research_retained is False
    assert entry.live_proven is False
    assert entry.capital_eligible is False
    assert entry.capital_reason == "research_demoted"
    assert entry.live_promotion_blocker == "weak_live_quality_and_contribution"
    assert entry.proposed_stake == pytest.approx(0.0)
    store.close()


def test_apply_allocation_rebalance_plan_updates_store(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.ML,
            definition={"model_ref": "m1"},
            stake=1.0,
            metadata={"oos_sharpe": 0.8},
        )
    )

    plan = build_allocation_rebalance_plan(
        store,
        metric="sharpe",
        min_observations=5,
        full_weight_observations=63,
        live_returns_for=lambda _hypothesis_id: [],
    )
    updates = apply_allocation_rebalance_plan(store, plan)
    updated = store.get("h1")

    assert "h1" in updates
    assert updated is not None
    assert updated.stake == pytest.approx(plan[0].proposed_stake)
    assert updated.metadata["lifecycle_research_retained"] is True
    assert updated.metadata["lifecycle_live_proven"] is False
    assert updated.metadata["lifecycle_capital_eligible"] is True
    assert updated.metadata["lifecycle_capital_reason"] == "research_backed"
    assert updated.metadata["lifecycle_rebalance_proposed_stake"] == pytest.approx(
        plan[0].proposed_stake
    )
    store.close()


def test_backfill_observation_returns_records_history_from_cached_signals(tmp_path):
    hdb = tmp_path / "hypotheses.db"
    sdb = tmp_path / "signal_cache.db"
    fdb = tmp_path / "forward_returns.db"

    store = HypothesisStore(hdb)
    store.register(
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.TECHNICAL,
            definition={"indicator": "roc_momentum", "params": {"window": 1}},
            stake=0.0,
        )
    )
    conn = sqlite3.connect(sdb)
    conn.execute(
        "CREATE TABLE signals (name TEXT, date TEXT, value REAL, resolution TEXT DEFAULT '1d', PRIMARY KEY (name, date))"
    )
    conn.executemany(
        "INSERT INTO signals (name, date, value, resolution) VALUES (?, ?, ?, ?)",
        [
            ("btc_ohlcv", "2026-03-20", 100.0, "1d"),
            ("btc_ohlcv", "2026-03-21", 101.0, "1d"),
            ("btc_ohlcv", "2026-03-22", 102.0, "1d"),
            ("btc_ohlcv", "2026-03-23", 103.0, "1d"),
        ],
    )
    conn.commit()
    conn.close()

    data_store = DataStore(sdb)
    forward_tracker = ForwardTracker(fdb)
    summary = backfill_observation_returns(
        hypothesis_store=store,
        data_store=data_store,
        forward_tracker=forward_tracker,
        asset="BTC",
        lookback_days=3,
    )

    records = forward_tracker.get_records("h1")
    assert summary.n_hypotheses == 1
    assert summary.n_days == 3
    assert summary.n_records == 3
    assert summary.n_failures == 0
    assert [record.date for record in records] == [
        "2026-03-21",
        "2026-03-22",
        "2026-03-23",
    ]

    forward_tracker.close()
    data_store.close()
    store.close()


def test_forward_tracker_get_realizable_returns_zeroes_short_side_in_long_only(tmp_path):
    tracker = ForwardTracker(tmp_path / "forward_returns.db")
    tracker.record("h1", "2026-03-21", 0.8, 0.016)
    tracker.record("h1", "2026-03-22", -0.7, 0.021)
    tracker.record("h1", "2026-03-23", 0.0, 0.0)

    assert tracker.get_returns("h1") == pytest.approx([0.016, 0.021, 0.0])
    assert tracker.get_realizable_returns("h1", supports_short=True) == pytest.approx(
        [0.016, 0.021, 0.0]
    )
    assert tracker.get_realizable_returns("h1", supports_short=False) == pytest.approx(
        [0.016, 0.0, 0.0]
    )

    tracker.close()


def test_apply_capital_redundancy_cap_caps_live_proven_duplicates():
    from alpha_os.hypotheses.breadth import apply_capital_redundancy_cap

    records = [
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.TECHNICAL,
            definition={"indicator": "roc_momentum", "params": {"window": 2}},
            stake=0.0,
        ),
        HypothesisRecord(
            hypothesis_id="h2",
            kind=HypothesisKind.TECHNICAL,
            definition={"indicator": "roc_momentum", "params": {"window": 2}},
            stake=0.0,
        ),
    ]
    plan = [
        AllocationRebalanceEntry(
            hypothesis_id="h1",
            current_stake=0.0,
            target_stake=0.4,
            proposed_stake=0.4,
            research_backed=False,
            research_retained=False,
            live_proven=True,
            capital_eligible=True,
            capital_reason="live_proven",
            live_promotion_blocker="eligible",
            n_observations=30,
            bootstrap_trust_value=0.0,
            blended_quality=0.4,
            live_quality=0.4,
            raw_live_quality=0.4,
            confidence=1.0,
            marginal_contribution=0.02,
        ),
        AllocationRebalanceEntry(
            hypothesis_id="h2",
            current_stake=0.0,
            target_stake=0.2,
            proposed_stake=0.2,
            research_backed=False,
            research_retained=False,
            live_proven=True,
            capital_eligible=True,
            capital_reason="live_proven",
            live_promotion_blocker="eligible",
            n_observations=30,
            bootstrap_trust_value=0.0,
            blended_quality=0.2,
            live_quality=0.2,
            raw_live_quality=0.2,
            confidence=1.0,
            marginal_contribution=0.01,
        ),
    ]
    data = {
        "btc_ohlcv": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0, 120.0],
    }

    capped = apply_capital_redundancy_cap(
        plan,
        records,
        data=data,
        asset="BTC",
        corr_max=0.7,
        floor=0.0,
    )

    kept = next(entry for entry in capped if entry.hypothesis_id == "h1")
    dropped = next(entry for entry in capped if entry.hypothesis_id == "h2")
    assert kept.proposed_stake == pytest.approx(0.4)
    assert dropped.proposed_stake == pytest.approx(0.0)
    assert dropped.redundancy_capped_by == "h1"
    assert dropped.redundancy_correlation > 0.7


def test_apply_live_proven_return_redundancy_cap_caps_positive_corr_cluster():
    from alpha_os.hypotheses.breadth import apply_live_proven_return_redundancy_cap

    plan = [
        AllocationRebalanceEntry(
            hypothesis_id="h1",
            current_stake=0.0,
            target_stake=0.4,
            proposed_stake=0.4,
            research_backed=False,
            research_retained=False,
            live_proven=True,
            capital_eligible=True,
            capital_reason="live_proven",
            live_promotion_blocker="eligible",
            n_observations=30,
            bootstrap_trust_value=0.0,
            blended_quality=0.4,
            live_quality=0.4,
            raw_live_quality=0.4,
            confidence=1.0,
            marginal_contribution=0.02,
        ),
        AllocationRebalanceEntry(
            hypothesis_id="h2",
            current_stake=0.0,
            target_stake=0.2,
            proposed_stake=0.2,
            research_backed=False,
            research_retained=False,
            live_proven=True,
            capital_eligible=True,
            capital_reason="live_proven",
            live_promotion_blocker="eligible",
            n_observations=30,
            bootstrap_trust_value=0.0,
            blended_quality=0.2,
            live_quality=0.2,
            raw_live_quality=0.2,
            confidence=1.0,
            marginal_contribution=0.01,
        ),
    ]

    returns = {
        "h1": [0.01, 0.02, 0.03, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.03],
        "h2": [0.01, 0.02, 0.03, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.03],
    }
    capped = apply_live_proven_return_redundancy_cap(
        plan,
        live_returns_for=lambda hypothesis_id: returns[hypothesis_id],
        corr_max=0.7,
        floor=0.0,
        min_observations=5,
    )

    kept = next(entry for entry in capped if entry.hypothesis_id == "h1")
    dropped = next(entry for entry in capped if entry.hypothesis_id == "h2")
    assert kept.proposed_stake == pytest.approx(0.4)
    assert dropped.proposed_stake == pytest.approx(0.0)
    assert dropped.redundancy_capped_by == "h1"
    assert dropped.redundancy_correlation > 0.7


def test_record_daily_contributions_persists_to_same_db(tmp_path):
    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.TECHNICAL,
            definition={"indicator": "x"},
            stake=1.0,
        )
    )
    store.register(
        HypothesisRecord(
            hypothesis_id="h2",
            kind=HypothesisKind.ML,
            definition={"model_ref": "m"},
            stake=1.0,
        )
    )

    contributions = record_daily_contributions(
        store,
        date="2026-03-21",
        predictions={"h1": 1.0, "h2": -1.0},
        realized_return=0.2,
    )

    rows_h1 = store.list_contributions("h1")
    rows_h2 = store.list_contributions("h2")

    assert "h1" in contributions
    assert "h2" in contributions
    assert len(rows_h1) == 1
    assert len(rows_h2) == 1
    assert rows_h1[0].date == "2026-03-21"
    assert rows_h2[0].date == "2026-03-21"
    store.close()
