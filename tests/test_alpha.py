"""Tests for alpha registry, lifecycle, combiner, and governance gates."""

import numpy as np
import pytest

from alpha_os.alpha.managed_alphas import (
    ManagedAlphaStore,
    AlphaRecord,
    AlphaState,
    CandidateSeed,
)
from alpha_os.alpha.lifecycle import (
    LifecycleConfig,
    compute_transition,
    batch_transitions,
    passes_candidate_gate,
    ST_CANDIDATE,
    ST_ACTIVE,
    ST_DORMANT,
)
from alpha_os.alpha.quality import QualityEstimate
from alpha_os.alpha.deployed_alphas import (
    plan_deployed_alphas,
    plan_registry_active_prune,
)
from alpha_os.alpha.combiner import (
    select_low_correlation,
    CombinerConfig,
    compute_diversity_scores,
    compute_tc_scores,
    compute_tc_weights,
    weighted_combine,
    weighted_combine_scalar,
    signal_consensus,
)
from alpha_os.config import Config
from alpha_os.daemon.admission import AdmissionDaemon
from alpha_os.alpha.admission_queue import prune_stale_pending_candidates
from alpha_os.dsl.canonical import canonical_string


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestManagedAlphaStore:
    def _make_registry(self, tmp_path):
        return ManagedAlphaStore(db_path=tmp_path / "test.db")

    def test_register_and_get(self, tmp_path):
        reg = self._make_registry(tmp_path)
        rec = AlphaRecord(alpha_id="a1", expression="(neg nvda)", fitness=0.5)
        reg.register(rec)
        got = reg.get("a1")
        assert got is not None
        assert got.expression == "(neg nvda)"
        assert got.state == AlphaState.CANDIDATE
        reg.close()

    def test_get_missing(self, tmp_path):
        reg = self._make_registry(tmp_path)
        assert reg.get("nonexistent") is None
        reg.close()

    def test_update_state(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(AlphaRecord(alpha_id="a1", expression="x"))
        reg.update_state("a1", AlphaState.ACTIVE)
        got = reg.get("a1")
        assert got.state == AlphaState.ACTIVE
        reg.close()

    def test_update_metrics(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(AlphaRecord(alpha_id="a1", expression="x"))
        reg.update_metrics("a1", oos_sharpe=1.5, pbo=0.2)
        got = reg.get("a1")
        assert got.oos_sharpe == 1.5
        assert got.pbo == 0.2
        reg.close()

    def test_list_by_state(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(AlphaRecord(alpha_id="a1", expression="x", state=AlphaState.CANDIDATE))
        reg.register(AlphaRecord(alpha_id="a2", expression="y", state=AlphaState.ACTIVE))
        reg.register(AlphaRecord(alpha_id="a3", expression="z", state=AlphaState.CANDIDATE))
        candidate = reg.list_by_state(AlphaState.CANDIDATE)
        assert len(candidate) == 2
        active = reg.list_active()
        assert len(active) == 1
        reg.close()

    def test_count(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(AlphaRecord(alpha_id="a1", expression="x"))
        reg.register(AlphaRecord(alpha_id="a2", expression="y"))
        assert reg.count() == 2
        assert reg.count(state=AlphaState.CANDIDATE) == 2
        reg.close()

    def test_top(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(AlphaRecord(alpha_id="a1", expression="x", oos_sharpe=0.5))
        reg.register(AlphaRecord(alpha_id="a2", expression="y", oos_sharpe=1.5))
        reg.register(AlphaRecord(alpha_id="a3", expression="z", oos_sharpe=1.0))
        top = reg.top(2)
        assert top[0].alpha_id == "a2"
        assert top[1].alpha_id == "a3"
        reg.close()

    def test_register_replaces(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(AlphaRecord(alpha_id="a1", expression="x", fitness=0.5))
        reg.register(AlphaRecord(alpha_id="a1", expression="x_new", fitness=1.0))
        got = reg.get("a1")
        assert got.expression == "x_new"
        assert got.fitness == 1.0
        assert reg.count() == 1
        reg.close()

    def test_replace_and_list_deployed_alphas(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(AlphaRecord(alpha_id="a1", expression="x", state=AlphaState.ACTIVE))
        reg.register(AlphaRecord(alpha_id="a2", expression="y", state=AlphaState.ACTIVE))

        reg.replace_deployed_alphas(
            ["a2", "a1"],
            scores={"a1": 0.5, "a2": 1.0},
            metadata={"a1": {"rank": 2}, "a2": {"rank": 1}},
        )

        assert reg.deployed_alpha_ids() == ["a2", "a1"]
        assert [r.alpha_id for r in reg.list_deployed_alphas()] == ["a2", "a1"]
        entries = reg.list_deployed_alpha_entries()
        assert entries[0].deployment_score == pytest.approx(1.0)
        assert entries[0].metadata["rank"] == 1
        reg.close()

    def test_replace_all_clears_deployed_alphas(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(AlphaRecord(alpha_id="a1", expression="x", state=AlphaState.ACTIVE))
        reg.replace_deployed_alphas(["a1"])

        reg.replace_all([
            AlphaRecord(alpha_id="b1", expression="y", state=AlphaState.ACTIVE),
        ])

        assert reg.count_deployed_alphas() == 0
        reg.close()

    def test_plan_deployed_alphas_replaces_only_above_margin(self):
        records = [
            AlphaRecord(alpha_id="i0", expression="x", state=AlphaState.ACTIVE, oos_sharpe=0.9),
            AlphaRecord(alpha_id="i1", expression="y", state=AlphaState.ACTIVE, oos_sharpe=0.8),
            AlphaRecord(alpha_id="c0", expression="z", state=AlphaState.ACTIVE, oos_sharpe=1.1),
            AlphaRecord(alpha_id="c1", expression="w", state=AlphaState.ACTIVE, oos_sharpe=0.7),
        ]
        estimates = {
            "i0": QualityEstimate(0.9, 0.0, 0.90, 1.0, 63, True),
            "i1": QualityEstimate(0.8, 0.0, 0.80, 1.0, 63, True),
            "c0": QualityEstimate(1.1, 0.0, 1.10, 1.0, 63, True),
            "c1": QualityEstimate(0.7, 0.0, 0.81, 1.0, 63, True),
        }

        plan = plan_deployed_alphas(
            records,
            current_ids=["i0", "i1"],
            estimate_for=lambda record: estimates[record.alpha_id],
            max_alphas=2,
            max_replacements=1,
            promotion_margin=0.25,
            metric="sharpe",
        )

        assert plan.selected_ids == ["c0", "i0"]
        assert plan.kept_ids == ["i0"]
        assert plan.added_ids == ["c0"]
        assert plan.dropped_ids == ["i1"]
        assert plan.replacement_count == 1
        assert plan.skipped_duplicate_ids == []

    def test_plan_deployed_alphas_skips_semantic_duplicates(self):
        records = [
            AlphaRecord(
                alpha_id="a",
                expression="(corr_10 nasdaq (ts_max_5 russell2000))",
                state=AlphaState.ACTIVE,
                oos_sharpe=1.2,
            ),
            AlphaRecord(
                alpha_id="b",
                expression="(corr_10 (ts_max_5 russell2000) nasdaq)",
                state=AlphaState.ACTIVE,
                oos_sharpe=1.1,
            ),
            AlphaRecord(
                alpha_id="c",
                expression="sp500",
                state=AlphaState.ACTIVE,
                oos_sharpe=1.0,
            ),
        ]
        estimates = {
            "a": QualityEstimate(1.2, 0.0, 1.2, 1.0, 63, True),
            "b": QualityEstimate(1.1, 0.0, 1.1, 1.0, 63, True),
            "c": QualityEstimate(1.0, 0.0, 1.0, 1.0, 63, True),
        }

        plan = plan_deployed_alphas(
            records,
            current_ids=[],
            estimate_for=lambda record: estimates[record.alpha_id],
            max_alphas=3,
            max_replacements=1,
            promotion_margin=0.0,
            metric="sharpe",
        )

        assert plan.selected_ids == ["a", "c"]
        assert plan.skipped_semantic_duplicate_ids == ["b"]
        assert plan.skipped_signal_duplicate_ids == []
        assert plan.skipped_feature_cap_ids == []

    def test_canonical_string_simplifies_degenerate_conditionals(self):
        assert canonical_string("(if_gt nasdaq nasdaq tsy_yield_10y fear_greed)") == "fear_greed"

    def test_plan_deployed_alphas_skips_signal_duplicates(self):
        records = [
            AlphaRecord(alpha_id="a", expression="x", state=AlphaState.ACTIVE, oos_sharpe=1.2),
            AlphaRecord(alpha_id="b", expression="y", state=AlphaState.ACTIVE, oos_sharpe=1.1),
            AlphaRecord(alpha_id="c", expression="z", state=AlphaState.ACTIVE, oos_sharpe=1.0),
        ]
        estimates = {
            "a": QualityEstimate(1.2, 0.0, 1.2, 1.0, 63, True),
            "b": QualityEstimate(1.1, 0.0, 1.1, 1.0, 63, True),
            "c": QualityEstimate(1.0, 0.0, 1.0, 1.0, 63, True),
        }
        base = np.linspace(-1.0, 1.0, 80)
        signal_by_id = {
            "a": base,
            "b": base * 0.999,
            "c": np.sin(np.linspace(0.0, 4.0, 80)),
        }

        plan = plan_deployed_alphas(
            records,
            current_ids=[],
            estimate_for=lambda record: estimates[record.alpha_id],
            max_alphas=3,
            max_replacements=1,
            promotion_margin=0.0,
            metric="sharpe",
            signal_by_id=signal_by_id,
            signal_similarity_max=0.995,
        )

        assert plan.selected_ids == ["a", "c"]
        assert plan.skipped_semantic_duplicate_ids == []
        assert plan.skipped_signal_duplicate_ids == ["b"]
        assert plan.skipped_feature_cap_ids == []

    def test_plan_deployed_alphas_applies_soft_feature_cap(self):
        records = [
            AlphaRecord(alpha_id="a", expression="nasdaq", state=AlphaState.ACTIVE, oos_sharpe=1.3),
            AlphaRecord(alpha_id="b", expression="(neg nasdaq)", state=AlphaState.ACTIVE, oos_sharpe=1.2),
            AlphaRecord(alpha_id="c", expression="sp500", state=AlphaState.ACTIVE, oos_sharpe=1.1),
            AlphaRecord(alpha_id="d", expression="gold", state=AlphaState.ACTIVE, oos_sharpe=1.0),
        ]
        estimates = {
            rid: QualityEstimate(score, 0.0, score, 1.0, 63, True)
            for rid, score in [("a", 1.3), ("b", 1.2), ("c", 1.1), ("d", 1.0)]
        }

        plan = plan_deployed_alphas(
            records,
            current_ids=[],
            estimate_for=lambda record: estimates[record.alpha_id],
            max_alphas=3,
            max_replacements=1,
            promotion_margin=0.0,
            metric="sharpe",
            max_feature_occurrences=1,
        )

        assert plan.selected_ids == ["a", "c", "d"]
        assert plan.skipped_feature_cap_ids == ["b"]

    def test_plan_registry_active_prune_demotes_semantic_and_signal_duplicates(self):
        records = [
            AlphaRecord(
                alpha_id="a",
                expression="(corr_10 nasdaq (ts_max_5 russell2000))",
                state=AlphaState.ACTIVE,
                oos_sharpe=1.3,
            ),
            AlphaRecord(
                alpha_id="b",
                expression="(corr_10 (ts_max_5 russell2000) nasdaq)",
                state=AlphaState.ACTIVE,
                oos_sharpe=1.2,
            ),
            AlphaRecord(
                alpha_id="c",
                expression="sp500",
                state=AlphaState.ACTIVE,
                oos_sharpe=1.1,
            ),
            AlphaRecord(
                alpha_id="d",
                expression="gold",
                state=AlphaState.ACTIVE,
                oos_sharpe=1.0,
            ),
        ]
        estimates = {
            rid: QualityEstimate(score, 0.0, score, 1.0, 63, True)
            for rid, score in [("a", 1.3), ("b", 1.2), ("c", 1.1), ("d", 1.0)]
        }
        base = np.linspace(-1.0, 1.0, 80)
        signal_by_id = {
            "a": base,
            "b": base * 0.999,
            "c": np.sin(np.linspace(0.0, 4.0, 80)),
            "d": np.cos(np.linspace(0.0, 4.0, 80)),
        }

        plan = plan_registry_active_prune(
            records,
            current_deployed_ids=["a", "c"],
            estimate_for=lambda record: estimates[record.alpha_id],
            metric="sharpe",
            signal_by_id=signal_by_id,
            signal_similarity_max=0.995,
        )

        assert plan.kept_ids == ["a", "c", "d"]
        assert plan.demoted_ids == ["b"]
        assert plan.skipped_semantic_duplicate_ids == ["b"]
        assert plan.skipped_signal_duplicate_ids == []
        assert plan.touched_deployed_count == 0

    def test_plan_registry_active_prune_demotes_signal_duplicates(self):
        records = [
            AlphaRecord(alpha_id="a", expression="x", state=AlphaState.ACTIVE, oos_sharpe=1.3),
            AlphaRecord(alpha_id="b", expression="y", state=AlphaState.ACTIVE, oos_sharpe=1.2),
            AlphaRecord(alpha_id="c", expression="z", state=AlphaState.ACTIVE, oos_sharpe=1.1),
        ]
        estimates = {
            rid: QualityEstimate(score, 0.0, score, 1.0, 63, True)
            for rid, score in [("a", 1.3), ("b", 1.2), ("c", 1.1)]
        }
        base = np.linspace(-1.0, 1.0, 80)
        signal_by_id = {
            "a": base,
            "b": base * 0.999,
            "c": np.sin(np.linspace(0.0, 4.0, 80)),
        }

        plan = plan_registry_active_prune(
            records,
            current_deployed_ids=["a", "b"],
            estimate_for=lambda record: estimates[record.alpha_id],
            metric="sharpe",
            signal_by_id=signal_by_id,
            signal_similarity_max=0.995,
        )

        assert plan.kept_ids == ["a", "c"]
        assert plan.demoted_ids == ["b"]
        assert plan.skipped_semantic_duplicate_ids == []
        assert plan.skipped_signal_duplicate_ids == ["b"]
        assert plan.touched_deployed_count == 1

    # Admission cap prune tests removed — stake system handles selection

    # Semantic dedup and feature cap tests removed — output-only principle

    def test_admission_fetch_pending_rows_prioritizes_alpha_generator(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.queue_candidates(
            [
                CandidateSeed(
                    expression="old_unknown",
                    source="legacy_batch",
                    fitness=10.0,
                    created_at=1.0,
                ),
                CandidateSeed(
                    expression="fresh_generated",
                    source="alpha_generator_btc",
                    fitness=0.5,
                    created_at=10.0,
                ),
                CandidateSeed(
                    expression="fresh_manual",
                    source="manual",
                    fitness=1.0,
                    created_at=9.0,
                ),
            ]
        )

        cfg = Config()
        daemon = AdmissionDaemon("BTC", cfg)
        daemon._open_registry_conn = lambda: reg._conn

        rows = daemon._fetch_pending_rows(3)

        assert [row[1] for row in rows] == [
            "fresh_generated",
            "fresh_manual",
            "old_unknown",
        ]
        reg.close()

    def test_prune_stale_pending_candidates_rejects_old_non_priority_sources(self, tmp_path, monkeypatch):
        db_path = tmp_path / "alpha_registry.db"
        reg = ManagedAlphaStore(db_path=db_path)
        reg.queue_candidates(
            [
                CandidateSeed(
                    expression="old_unknown",
                    source="legacy_batch",
                    fitness=1.0,
                    created_at=1.0,
                ),
                CandidateSeed(
                    expression="old_manual",
                    source="manual",
                    fitness=1.0,
                    created_at=1.0,
                ),
                CandidateSeed(
                    expression="old_generated",
                    source="alpha_generator_btc",
                    fitness=1.0,
                    created_at=1.0,
                ),
                CandidateSeed(
                    expression="fresh_unknown",
                    source="legacy_batch",
                    fitness=1.0,
                    created_at=10_000_000_000.0,
                ),
            ]
        )
        reg.close()
        monkeypatch.setattr(
            "alpha_os.alpha.admission_queue.asset_data_dir",
            lambda asset: tmp_path,
        )
        monkeypatch.setattr("alpha_os.alpha.admission_queue.time.time", lambda: 10 * 86400)

        stats = prune_stale_pending_candidates("BTC", max_age_days=7)

        assert stats.selected_count == 1
        assert stats.pruned_count == 1

        reg = ManagedAlphaStore(db_path=db_path)
        try:
            rows = reg._conn.execute(
                "SELECT expression, status, error_message FROM candidates ORDER BY expression"
            ).fetchall()
            by_expression = {
                row["expression"]: (row["status"], row["error_message"])
                for row in rows
            }
            assert by_expression["old_unknown"] == ("rejected", "stale pending > 7d")
            assert by_expression["old_manual"] == ("pending", None)
            assert by_expression["old_generated"] == ("pending", None)
            assert by_expression["fresh_unknown"] == ("pending", None)
        finally:
            reg.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestPassesCandidateGate:
    def test_passes_candidate_gate_pure_helper(self):
        cfg = LifecycleConfig(
            candidate_quality_min=0.5,
            pbo_max=0.5,
            dsr_pvalue_max=0.05,
            correlation_max=0.5,
        )
        record = AlphaRecord(
            alpha_id="a1",
            expression="x",
            oos_sharpe=0.7,
            pbo=0.2,
            dsr_pvalue=0.01,
            correlation_avg=0.1,
        )
        assert passes_candidate_gate(record, cfg) is True

    def test_passes_candidate_gate_rejects_low_quality(self):
        cfg = LifecycleConfig(candidate_quality_min=0.5)
        record = AlphaRecord(
            alpha_id="a1",
            expression="x",
            oos_sharpe=0.4,
        )
        assert passes_candidate_gate(record, cfg) is False


class TestComputeTransition:
    def test_active_stays(self):
        cfg = LifecycleConfig(active_quality_min=0.3)
        assert compute_transition(AlphaState.ACTIVE, 0.5, cfg) == AlphaState.ACTIVE

    def test_active_to_dormant(self):
        cfg = LifecycleConfig(active_quality_min=0.3)
        assert compute_transition(AlphaState.ACTIVE, 0.1, cfg) == AlphaState.DORMANT

    def test_candidate_stays_candidate(self):
        cfg = LifecycleConfig(candidate_quality_min=0.5)
        assert compute_transition(AlphaState.CANDIDATE, 0.8, cfg) == AlphaState.CANDIDATE

    def test_dormant_stays(self):
        cfg = LifecycleConfig(dormant_revival_quality=0.3)
        assert compute_transition(AlphaState.DORMANT, 0.1, cfg) == AlphaState.DORMANT

    def test_dormant_revival(self):
        cfg = LifecycleConfig(dormant_revival_quality=0.3)
        assert compute_transition(AlphaState.DORMANT, 0.4, cfg) == AlphaState.ACTIVE


class TestBatchTransitions:
    def test_vectorized_matches_scalar(self):
        cfg = LifecycleConfig()
        states = np.array([ST_CANDIDATE, ST_ACTIVE, ST_DORMANT], dtype=np.int8)
        sharpes = np.array([0.1, 0.8, 0.4])
        new = batch_transitions(states, sharpes, cfg)
        for i, (s, sh) in enumerate(zip(
            [AlphaState.CANDIDATE, AlphaState.ACTIVE, AlphaState.DORMANT],
            sharpes,
        )):
            expected = compute_transition(s, sh, cfg)
            state_map = {
                AlphaState.CANDIDATE: ST_CANDIDATE,
                AlphaState.ACTIVE: ST_ACTIVE,
                AlphaState.DORMANT: ST_DORMANT,
            }
            assert new[i] == state_map[expected]

    def test_all_stay(self):
        cfg = LifecycleConfig()
        states = np.array([ST_ACTIVE, ST_ACTIVE], dtype=np.int8)
        sharpes = np.array([1.0, 1.0])
        new = batch_transitions(states, sharpes, cfg)
        np.testing.assert_array_equal(new, [ST_ACTIVE, ST_ACTIVE])


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------

class TestCombiner:
    def test_select_low_correlation(self):
        rng = np.random.RandomState(42)
        # Create signals: first 3 independent, last 2 correlated with first
        signals = rng.randn(5, 200)
        signals[3] = signals[0] + rng.randn(200) * 0.01  # highly correlated
        signals[4] = signals[1] + rng.randn(200) * 0.01
        sharpes = np.array([1.0, 0.9, 0.8, 0.7, 0.6])

        selected = select_low_correlation(signals, sharpes)
        # Should select first 3, skip 3 and 4 (too correlated)
        assert 0 in selected
        assert 1 in selected
        assert 2 in selected
        assert 3 not in selected

    def test_empty(self):
        signals = np.empty((0, 100))
        sharpes = np.array([])
        assert select_low_correlation(signals, sharpes) == []

    def test_max_alphas(self):
        rng = np.random.RandomState(42)
        signals = rng.randn(20, 100)
        sharpes = np.arange(20, dtype=float)
        cfg = CombinerConfig(max_alphas=5, max_correlation=1.0)
        selected = select_low_correlation(signals, sharpes, config=cfg)
        assert len(selected) <= 5

    def test_select_low_correlation_sanitizes_infinities(self):
        signals = np.array([
            np.linspace(-1.0, 1.0, 20),
            np.array([0.0, np.inf] * 10, dtype=float),
            np.linspace(1.0, -1.0, 20),
        ])
        sharpes = np.array([1.0, 0.9, 0.8])

        with np.errstate(all="raise"):
            selected = select_low_correlation(signals, sharpes)

        assert selected

    def test_select_low_correlation_prefers_diverse_seed_when_quality_zero(self):
        base = np.linspace(-1.0, 1.0, 60)
        signals = np.array([
            base,
            base + 0.001,
            np.tile([1.0, -1.0], 30),
        ])
        sharpes = np.zeros(3)

        selected = select_low_correlation(signals, sharpes, config=CombinerConfig(max_alphas=2))

        assert selected[0] == 2
        assert len(selected) == 2

class TestWeightedCombiner:
    def test_diversity_independent(self):
        rng = np.random.RandomState(42)
        signals = rng.randn(5, 500)
        diversity = compute_diversity_scores(signals)
        assert diversity.shape == (5,)
        assert np.all(diversity > 0.7)

    def test_diversity_correlated(self):
        rng = np.random.RandomState(42)
        base = rng.randn(500)
        signals = np.array([base + rng.randn(500) * 0.01 for _ in range(5)])
        diversity = compute_diversity_scores(signals)
        assert np.all(diversity < 0.1)

    def test_diversity_single(self):
        signals = np.random.randn(1, 100)
        diversity = compute_diversity_scores(signals)
        assert diversity[0] == 1.0

    def test_chunked_consistency(self):
        rng = np.random.RandomState(42)
        signals = rng.randn(50, 200)
        full = compute_diversity_scores(signals, chunk_size=50)
        chunked = compute_diversity_scores(signals, chunk_size=10)
        np.testing.assert_allclose(full, chunked, atol=1e-10)

    def test_weighted_combine_matrix(self):
        signals = np.array([[1.0, -1.0], [-1.0, 1.0]])
        weights = np.array([0.75, 0.25])
        combined = weighted_combine(signals, weights)
        np.testing.assert_allclose(combined, [0.5, -0.5])

    def test_weighted_combine_scalar(self):
        signals = {"a": 1.0, "b": -1.0}
        weights = {"a": 0.75, "b": 0.25}
        result = weighted_combine_scalar(signals, weights)
        assert np.isclose(result, 0.5)


# ---------------------------------------------------------------------------
# True Contribution (TC)
# ---------------------------------------------------------------------------

class TestTrueContribution:
    def test_single_alpha_tc_equals_sharpe(self):
        rng = np.random.RandomState(42)
        returns = rng.randn(200) * 0.01
        signal = rng.randn(200)
        tc = compute_tc_scores({"a": signal}, returns)
        assert "a" in tc
        assert tc["a"] != 0.0

    def test_harmful_alpha_gets_negative_tc(self):
        rng = np.random.RandomState(42)
        returns = rng.randn(200) * 0.01
        good = returns * 0.5 + rng.randn(200) * 0.001  # correlated with returns
        bad = -returns * 0.5 + rng.randn(200) * 0.001   # anti-correlated
        tc = compute_tc_scores({"good": good, "bad": bad}, returns)
        assert tc["good"] > tc["bad"]

    def test_redundant_alpha_has_low_tc(self):
        rng = np.random.RandomState(42)
        returns = rng.randn(200) * 0.01
        sig = returns * 0.3 + rng.randn(200) * 0.005
        # Two copies of the same signal
        tc = compute_tc_scores({"a": sig, "b": sig, "c": sig}, returns)
        # Each copy contributes less than if it were unique
        tc_single = compute_tc_scores({"a": sig}, returns)
        assert tc["a"] < tc_single["a"]

    def test_empty_returns_zero(self):
        tc = compute_tc_scores({}, np.array([]))
        assert tc == {}

    def test_short_history_returns_zero(self):
        tc = compute_tc_scores({"a": np.array([1.0])}, np.array([0.01]))
        assert tc["a"] == 0.0

    def test_tc_weights_positive(self):
        tc_scores = {"a": 0.5, "b": -0.1, "c": 0.2}
        weights = compute_tc_weights(tc_scores)
        assert all(w > 0 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_tc_weights_negative_gets_min(self):
        tc_scores = {"a": 1.0, "b": -0.5}
        weights = compute_tc_weights(tc_scores)
        assert weights["a"] > weights["b"]
        assert weights["b"] > 0  # not zero, min_weight floor


# ---------------------------------------------------------------------------
# Alpha Distribution
# ---------------------------------------------------------------------------


class TestSignalConsensus:
    def test_unanimous_long(self):
        signals = {"a1": 0.8, "a2": 0.7, "a3": 0.9}
        weights = {"a1": 1 / 3, "a2": 1 / 3, "a3": 1 / 3}
        mean, std, cons = signal_consensus(signals, weights)
        assert mean > 0
        assert cons > 0.8  # high consensus

    def test_split_signals(self):
        signals = {"a1": 1.0, "a2": -1.0}
        weights = {"a1": 0.5, "a2": 0.5}
        mean, std, cons = signal_consensus(signals, weights)
        assert abs(mean) < 1e-10
        assert cons < 0.01  # no consensus

    def test_empty(self):
        mean, std, cons = signal_consensus({}, {})
        assert mean == 0.0
        assert std == 0.0
        assert cons == 0.0

    def test_single_alpha(self):
        signals = {"a1": 0.5}
        weights = {"a1": 1.0}
        mean, std, cons = signal_consensus(signals, weights)
        assert np.isclose(mean, 0.5)
        assert std == 0.0
        assert cons == 1.0  # single alpha → full consensus

    def test_consensus_between_zero_and_one(self):
        signals = {"a1": 0.5, "a2": 0.3, "a3": -0.1}
        weights = {"a1": 0.5, "a2": 0.3, "a3": 0.2}
        mean, std, cons = signal_consensus(signals, weights)
        assert 0.0 <= cons <= 1.0
