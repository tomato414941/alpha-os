"""Tests for the evolution module (GP, archive, behavior)."""
import numpy as np
import pytest

from alpha_os.dsl.expr import Feature, UnaryOp, BinaryOp, RollingOp, Constant, ConditionalOp, LagOp
from alpha_os.evolution.gp import (
    GPConfig,
    GPEvolver,
    _tree_depth,
    _node_count,
    _ast_signature,
    _jaccard_similarity,
)
from alpha_os.evolution.archive import AlphaArchive, ArchiveConfig, passes_sanity_filter
from alpha_os.evolution.behavior import (
    compute_behavior,
    _feature_bucket,
    _holding_half_life,
)


FEATURES = ["f1", "f2", "f3"]


# ---------------------------------------------------------------------------
# GP tests
# ---------------------------------------------------------------------------

class TestTreeDepth:
    def test_leaf(self):
        assert _tree_depth(Feature("x")) == 0

    def test_constant(self):
        assert _tree_depth(Constant(1.0)) == 0

    def test_unary(self):
        expr = UnaryOp("neg", Feature("x"))
        assert _tree_depth(expr) == 1

    def test_binary(self):
        expr = BinaryOp("add", Feature("x"), Feature("y"))
        assert _tree_depth(expr) == 1

    def test_rolling(self):
        expr = RollingOp("mean", 10, Feature("x"))
        assert _tree_depth(expr) == 1

    def test_nested(self):
        expr = BinaryOp(
            "sub",
            RollingOp("roc", 10, Feature("x")),
            RollingOp("roc", 10, Feature("y")),
        )
        assert _tree_depth(expr) == 2

    def test_lag(self):
        assert _tree_depth(LagOp("lag", 10, Feature("x"))) == 1

    def test_conditional_flat(self):
        expr = ConditionalOp(
            "if_gt", Feature("x"), Feature("y"), Feature("z"), Feature("w")
        )
        assert _tree_depth(expr) == 1

    def test_conditional_nested(self):
        expr = ConditionalOp(
            "if_gt",
            RollingOp("mean", 10, Feature("x")),
            Feature("y"),
            BinaryOp("add", Feature("z"), Feature("w")),
            UnaryOp("neg", Feature("a")),
        )
        assert _tree_depth(expr) == 2


class TestNodeCount:
    def test_leaf(self):
        assert _node_count(Feature("x")) == 1

    def test_binary(self):
        expr = BinaryOp("add", Feature("x"), Feature("y"))
        assert _node_count(expr) == 3

    def test_nested(self):
        expr = BinaryOp(
            "sub",
            RollingOp("roc", 10, Feature("x")),
            Feature("y"),
        )
        assert _node_count(expr) == 4

    def test_lag(self):
        assert _node_count(LagOp("lag", 10, Feature("x"))) == 2

    def test_conditional(self):
        expr = ConditionalOp(
            "if_gt", Feature("x"), Feature("y"), Feature("z"), Feature("w")
        )
        assert _node_count(expr) == 5


class TestAstSimilarity:
    def test_signature_not_empty(self):
        expr = BinaryOp("add", Feature("f1"), Feature("f2"))
        sig = _ast_signature(expr)
        assert len(sig) > 0

    def test_jaccard_identical_is_one(self):
        a = _ast_signature(BinaryOp("add", Feature("f1"), Feature("f2")))
        b = _ast_signature(BinaryOp("add", Feature("f1"), Feature("f2")))
        assert _jaccard_similarity(a, b) == pytest.approx(1.0)

    def test_jaccard_different_is_lower(self):
        a = _ast_signature(BinaryOp("add", Feature("f1"), Feature("f2")))
        b = _ast_signature(UnaryOp("neg", Feature("f3")))
        assert _jaccard_similarity(a, b) < 1.0


class TestGPEvolver:
    @staticmethod
    def _make_evaluate_fn():
        rng = np.random.RandomState(42)
        data = {f"f{i}": rng.randn(100) for i in range(1, 4)}

        def evaluate(expr):
            try:
                signal = expr.evaluate(data)
                signal = np.nan_to_num(signal, nan=0.0)
                if np.std(signal) == 0:
                    return 0.0
                return float(np.abs(np.mean(signal)) / (np.std(signal) + 1e-10))
            except Exception:
                return -999.0

        return evaluate

    def test_run_returns_results(self):
        cfg = GPConfig(pop_size=20, n_generations=3, max_depth=2)
        evolver = GPEvolver(FEATURES, self._make_evaluate_fn(), config=cfg, seed=42)
        results = evolver.run()
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_results_sorted_descending(self):
        cfg = GPConfig(pop_size=20, n_generations=3, max_depth=2)
        evolver = GPEvolver(FEATURES, self._make_evaluate_fn(), config=cfg, seed=42)
        results = evolver.run()
        fitnesses = [f for _, f in results]
        for i in range(len(fitnesses) - 1):
            assert fitnesses[i] >= fitnesses[i + 1]

    def test_results_deduplicated(self):
        cfg = GPConfig(pop_size=20, n_generations=3, max_depth=2)
        evolver = GPEvolver(FEATURES, self._make_evaluate_fn(), config=cfg, seed=42)
        results = evolver.run()
        reprs = [repr(e) for e, _ in results]
        assert len(reprs) == len(set(reprs))

    def test_bloat_penalty(self):
        cfg = GPConfig(pop_size=10, n_generations=1, bloat_penalty=100.0)
        evolver = GPEvolver(FEATURES, self._make_evaluate_fn(), config=cfg, seed=42)
        results = evolver.run()
        for _, fit in results:
            assert fit < 100.0  # penalty should dominate

    def test_depth_penalty(self):
        cfg = GPConfig(
            pop_size=10,
            n_generations=1,
            bloat_penalty=0.0,
            depth_penalty=10.0,
            max_depth=3,
        )
        evolver = GPEvolver(FEATURES, self._make_evaluate_fn(), config=cfg, seed=42)
        results = evolver.run()
        assert len(results) > 0

    def test_similarity_penalty_uses_memory(self):
        cfg = GPConfig(pop_size=10, n_generations=1, similarity_penalty=1.0, max_depth=2)
        evolver = GPEvolver(FEATURES, self._make_evaluate_fn(), config=cfg, seed=42)
        expr = BinaryOp("add", Feature("f1"), Feature("f2"))
        evolver._update_signature_memory([expr])
        fit = evolver._evaluate_batch([expr])[0]
        raw = self._make_evaluate_fn()(expr)
        assert fit < raw

    def test_custom_generator(self):
        from alpha_os.dsl.generator import AlphaGenerator
        gen = AlphaGenerator.with_random_subset(FEATURES, k=2, seed=42)
        cfg = GPConfig(pop_size=10, n_generations=1, max_depth=2)
        evolver = GPEvolver(FEATURES, self._make_evaluate_fn(), config=cfg, seed=42, generator=gen)
        results = evolver.run()
        assert len(results) > 0
        # All expressions should only use subset features
        for expr, _ in results:
            from alpha_os.dsl import collect_feature_names
            used = collect_feature_names(expr)
            assert used.issubset(gen.feature_subset)


# ---------------------------------------------------------------------------
# Archive tests
# ---------------------------------------------------------------------------

class TestAlphaArchive:
    def test_add_new_cell(self):
        archive = AlphaArchive()
        expr = Feature("f1")
        behavior = np.array([50.0, 10.0, 5.0])
        assert archive.add(expr, 0.5, behavior) is True
        assert archive.size == 1

    def test_add_better_replaces(self):
        archive = AlphaArchive()
        behavior = np.array([50.0, 10.0, 5.0])
        archive.add(Feature("f1"), 0.3, behavior)
        assert archive.add(Feature("f2"), 0.7, behavior) is True
        assert archive.size == 1
        best = archive.best(1)
        assert repr(best[0][0]) == "f2"

    def test_add_worse_rejected(self):
        archive = AlphaArchive()
        behavior = np.array([50.0, 10.0, 5.0])
        archive.add(Feature("f1"), 0.7, behavior)
        assert archive.add(Feature("f2"), 0.3, behavior) is False
        assert archive.size == 1

    def test_different_cells(self):
        archive = AlphaArchive()
        archive.add(Feature("f1"), 0.5, np.array([10.0, 5.0, 3.0]))
        archive.add(Feature("f2"), 0.6, np.array([90.0, 80.0, 15.0]))
        assert archive.size == 2

    def test_capacity(self):
        cfg = ArchiveConfig(dims=(5, 5, 5))
        archive = AlphaArchive(config=cfg)
        assert archive.capacity == 125

    def test_coverage(self):
        cfg = ArchiveConfig(dims=(2, 2, 2))
        archive = AlphaArchive(config=cfg)
        archive.add(Feature("f1"), 0.5, np.array([20.0, 10.0, 5.0]))
        assert archive.coverage == 1 / 8

    def test_best_ordering(self):
        archive = AlphaArchive()
        archive.add(Feature("f1"), 0.3, np.array([10.0, 5.0, 3.0]))
        archive.add(Feature("f2"), 0.9, np.array([50.0, 50.0, 10.0]))
        archive.add(Feature("f3"), 0.6, np.array([90.0, 80.0, 15.0]))
        best = archive.best(3)
        assert best[0][1] >= best[1][1] >= best[2][1]

    def test_sample(self):
        archive = AlphaArchive()
        for i in range(5):
            archive.add(
                Feature(f"f{i}"),
                float(i),
                np.array([i * 10.0, i * 10.0, i + 1.0]),
            )
        import random

        sampled = archive.sample(3, rng=random.Random(42))
        assert len(sampled) == 3

    def test_elites(self):
        archive = AlphaArchive()
        archive.add(Feature("f1"), 0.5, np.array([50.0, 10.0, 5.0]))
        elites = archive.elites()
        assert len(elites) == 1
        assert elites[0][1] == 0.5

    def test_boundary_clipping(self):
        archive = AlphaArchive()
        behavior = np.array([200.0, 200.0, 50.0])  # all out of range
        assert archive.add(Feature("f1"), 0.5, behavior) is True

    def test_save_load_roundtrip(self, tmp_path):
        archive = AlphaArchive()
        archive.add(Feature("f1"), 0.5, np.array([50.0, 10.0, 5.0]))
        archive.add(Feature("f2"), 0.8, np.array([20.0, 30.0, 3.0]))
        db_path = tmp_path / "archive.db"
        n = archive.save_to_db(db_path)
        assert n == 2

        loaded = AlphaArchive.load_from_db(db_path)
        assert loaded.size == 2
        best = loaded.best(2)
        assert best[0][1] == 0.8  # f2 has higher fitness

    def test_load_nonexistent(self, tmp_path):
        loaded = AlphaArchive.load_from_db(tmp_path / "missing.db")
        assert loaded.size == 0

    def test_save_empty_archive(self, tmp_path):
        archive = AlphaArchive()
        db_path = tmp_path / "archive.db"
        n = archive.save_to_db(db_path)
        assert n == 0
        loaded = AlphaArchive.load_from_db(db_path)
        assert loaded.size == 0

    def test_save_load_preserves_cells(self, tmp_path):
        archive = AlphaArchive()
        b1 = np.array([10.0, 5.0, 3.0])
        b2 = np.array([90.0, 80.0, 15.0])
        archive.add(Feature("f1"), 0.5, b1)
        archive.add(Feature("f2"), 0.8, b2)

        cell1 = archive._to_cell(b1)
        cell2 = archive._to_cell(b2)
        assert cell1 != cell2

        db_path = tmp_path / "archive.db"
        archive.save_to_db(db_path)
        loaded = AlphaArchive.load_from_db(db_path)
        assert loaded.size == 2
        assert cell1 in loaded._grid
        assert cell2 in loaded._grid

    def test_save_load_complex_expr(self, tmp_path):
        archive = AlphaArchive()
        expr = BinaryOp("add", RollingOp("mean", 10, Feature("f1")), Feature("f2"))
        archive.add(expr, 1.0, np.array([50.0, 10.0, 4.0]))

        db_path = tmp_path / "archive.db"
        archive.save_to_db(db_path)
        loaded = AlphaArchive.load_from_db(db_path)
        assert loaded.size == 1
        loaded_expr = loaded.best(1)[0][0]
        assert repr(loaded_expr) == repr(expr)


class TestSanityFilter:
    def test_passes_normal_signal(self):
        signal = np.random.randn(100)
        assert passes_sanity_filter(signal) is True

    def test_rejects_constant(self):
        assert passes_sanity_filter(np.ones(100)) is False

    def test_rejects_high_nan(self):
        signal = np.full(100, np.nan)
        signal[:5] = np.random.randn(5)
        assert passes_sanity_filter(signal) is False

    def test_rejects_inf(self):
        signal = np.random.randn(100)
        signal[50] = np.inf
        assert passes_sanity_filter(signal) is False

    def test_rejects_empty(self):
        assert passes_sanity_filter(np.array([])) is False

    def test_accepts_low_nan(self):
        signal = np.random.randn(100)
        signal[:5] = np.nan  # 5% NaN
        assert passes_sanity_filter(signal) is True


class TestAddIfEmpty:
    def test_adds_to_empty_cell(self):
        archive = AlphaArchive()
        signal = np.random.randn(100)
        behavior = np.array([50.0, 10.0, 5.0])
        assert archive.add_if_empty(Feature("f1"), behavior, signal) is True
        assert archive.size == 1

    def test_rejects_occupied_cell(self):
        archive = AlphaArchive()
        signal = np.random.randn(100)
        behavior = np.array([50.0, 10.0, 5.0])
        archive.add_if_empty(Feature("f1"), behavior, signal)
        assert archive.add_if_empty(Feature("f2"), behavior, signal) is False
        assert archive.size == 1

    def test_rejects_bad_signal(self):
        archive = AlphaArchive()
        signal = np.ones(100)  # constant → fails sanity
        behavior = np.array([50.0, 10.0, 5.0])
        assert archive.add_if_empty(Feature("f1"), behavior, signal) is False
        assert archive.size == 0


# ---------------------------------------------------------------------------
# Behavior tests
# ---------------------------------------------------------------------------

class TestBehavior:
    def test_compute_behavior_shape(self):
        signal = np.random.randn(100)
        expr = Feature("f1")
        b = compute_behavior(signal, expr)
        assert b.shape == (3,)

    def test_feature_bucket_none(self):
        assert _feature_bucket(None) == 0

    def test_feature_bucket_deterministic(self):
        subset = frozenset(["f1", "f3", "f7"])
        b1 = _feature_bucket(subset)
        b2 = _feature_bucket(subset)
        assert b1 == b2
        assert 0 <= b1 < 100

    def test_feature_bucket_different_subsets(self):
        s1 = frozenset(["f1", "f2"])
        s2 = frozenset(["f3", "f4"])
        # Different subsets should (usually) get different buckets
        # Not guaranteed but extremely likely with mod 100
        b1 = _feature_bucket(s1)
        b2 = _feature_bucket(s2)
        assert 0 <= b1 < 100
        assert 0 <= b2 < 100

    def test_holding_half_life_constant(self):
        signal = np.ones(100)
        assert _holding_half_life(signal) == 0.0

    def test_holding_half_life_positive(self):
        rng = np.random.RandomState(42)
        signal = np.zeros(200)
        for i in range(1, 200):
            signal[i] = 0.95 * signal[i - 1] + rng.randn() * 0.1
        hl = _holding_half_life(signal)
        assert hl > 5.0

    def test_complexity_is_node_count(self):
        expr = BinaryOp("add", Feature("f1"), Feature("f2"))
        signal = np.random.randn(100)
        b = compute_behavior(signal, expr)
        assert b[2] == 3.0  # binary + 2 leaves

    def test_behavior_with_feature_subset(self):
        expr = Feature("f1")
        signal = np.random.randn(100)
        subset = frozenset(["f1", "f5", "f10"])
        b = compute_behavior(signal, expr, feature_subset=subset)
        assert b[0] == float(_feature_bucket(subset))
        assert b[2] == 1.0  # single feature node
