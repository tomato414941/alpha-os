"""Tests for the evolution module (GP, discovery pool, behavior)."""
import json
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
from alpha_os.evolution.discovery_pool import DiscoveryPool, DiscoveryPoolConfig, passes_sanity_filter
from alpha_os.evolution.behavior import (
    compute_behavior,
    _persistence,
    _activity,
    _price_beta,
    _vol_sensitivity,
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

class TestDiscoveryPool:
    def _b(self, p=25.0, a=0.5, pb=0.0, vs=0.0):
        """Helper: build a 4D behavior vector."""
        return np.array([p, a, pb, vs])

    def test_add_new_cell(self):
        pool = DiscoveryPool()
        assert pool.add(Feature("f1"), 0.5, self._b()) is True
        assert pool.size == 1

    def test_add_better_replaces(self):
        pool = DiscoveryPool()
        b = self._b()
        pool.add(Feature("f1"), 0.3, b)
        assert pool.add(Feature("f2"), 0.7, b) is True
        assert pool.size == 1
        assert repr(pool.best(1)[0][0]) == "f2"

    def test_add_worse_rejected(self):
        pool = DiscoveryPool()
        b = self._b()
        pool.add(Feature("f1"), 0.7, b)
        assert pool.add(Feature("f2"), 0.3, b) is False
        assert pool.size == 1

    def test_different_cells(self):
        pool = DiscoveryPool()
        pool.add(Feature("f1"), 0.5, self._b(5.0, 0.2, -0.5, -0.3))
        pool.add(Feature("f2"), 0.6, self._b(40.0, 0.9, 0.7, 0.5))
        assert pool.size == 2

    def test_capacity(self):
        cfg = DiscoveryPoolConfig(dims=(5, 5, 5, 5))
        pool = DiscoveryPool(config=cfg)
        assert pool.capacity == 625

    def test_coverage(self):
        cfg = DiscoveryPoolConfig(dims=(2, 2, 2, 2))
        pool = DiscoveryPool(config=cfg)
        pool.add(Feature("f1"), 0.5, self._b())
        assert pool.coverage == 1 / 16

    def test_best_ordering(self):
        pool = DiscoveryPool()
        pool.add(Feature("f1"), 0.3, self._b(5.0, 0.2, -0.5, -0.3))
        pool.add(Feature("f2"), 0.9, self._b(25.0, 0.5, 0.0, 0.0))
        pool.add(Feature("f3"), 0.6, self._b(40.0, 0.9, 0.7, 0.5))
        best = pool.best(3)
        assert best[0][1] >= best[1][1] >= best[2][1]

    def test_sample(self):
        pool = DiscoveryPool()
        for i in range(5):
            pool.add(
                Feature(f"f{i}"),
                float(i),
                self._b(i * 8.0, i * 0.15, i * 0.2 - 0.5, i * 0.1 - 0.3),
            )
        import random
        sampled = pool.sample(3, rng=random.Random(42))
        assert len(sampled) == 3

    def test_elites(self):
        pool = DiscoveryPool()
        pool.add(Feature("f1"), 0.5, self._b())
        elites = pool.elites()
        assert len(elites) == 1
        assert elites[0][1] == 0.5

    def test_boundary_clipping(self):
        pool = DiscoveryPool()
        behavior = np.array([200.0, 5.0, 5.0, 5.0])  # out of range
        assert pool.add(Feature("f1"), 0.5, behavior) is True

    def test_save_load_roundtrip(self, tmp_path):
        pool = DiscoveryPool()
        pool.add(Feature("f1"), 0.5, self._b(10.0, 0.3, -0.2, 0.1))
        pool.add(Feature("f2"), 0.8, self._b(30.0, 0.7, 0.4, -0.3))
        db_path = tmp_path / "discovery_pool.db"
        n = pool.save_to_db(db_path)
        assert n == 2
        loaded = DiscoveryPool.load_from_db(db_path)
        assert loaded.size == 2
        assert loaded.best(2)[0][1] == 0.8

    def test_load_nonexistent(self, tmp_path):
        loaded = DiscoveryPool.load_from_db(tmp_path / "missing.db")
        assert loaded.size == 0

    def test_save_empty_pool(self, tmp_path):
        pool = DiscoveryPool()
        db_path = tmp_path / "discovery_pool.db"
        n = pool.save_to_db(db_path)
        assert n == 0
        loaded = DiscoveryPool.load_from_db(db_path)
        assert loaded.size == 0

    def test_save_load_preserves_entries(self, tmp_path):
        pool = DiscoveryPool()
        pool.add(Feature("f1"), 0.5, self._b(5.0, 0.2, -0.3, 0.1))
        pool.add(Feature("f2"), 0.8, self._b(40.0, 0.8, 0.6, -0.5))
        db_path = tmp_path / "discovery_pool.db"
        pool.save_to_db(db_path)
        loaded = DiscoveryPool.load_from_db(db_path)
        assert loaded.size == 2
        exprs = {repr(e) for e, _ in loaded.best(10)}
        assert repr(Feature("f1")) in exprs
        assert repr(Feature("f2")) in exprs

    def test_save_load_complex_expr(self, tmp_path):
        pool = DiscoveryPool()
        expr = BinaryOp("add", RollingOp("mean", 10, Feature("f1")), Feature("f2"))
        pool.add(expr, 1.0, self._b(25.0, 0.6, 0.1, -0.1))
        db_path = tmp_path / "discovery_pool.db"
        pool.save_to_db(db_path)
        loaded = DiscoveryPool.load_from_db(db_path)
        assert loaded.size == 1
        assert repr(loaded.best(1)[0][0]) == repr(expr)

    def test_load_skips_dim_mismatch(self, tmp_path):
        """Old 3D entries are skipped when grid expects 4D."""
        pool = DiscoveryPool()
        pool.add(Feature("f1"), 0.5, self._b())
        db_path = tmp_path / "discovery_pool.db"
        pool.save_to_db(db_path)
        # Tamper: rewrite with 3D behavior
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE discovery_pool SET behavior = ? WHERE 1=1",
            (json.dumps([10.0, 5.0, 3.0]),),
        )
        conn.commit()
        conn.close()
        loaded = DiscoveryPool.load_from_db(db_path)
        assert loaded.size == 0


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


class TestStoreCandidate:
    def test_adds_to_empty_cell(self):
        pool = DiscoveryPool()
        signal = np.random.randn(100)
        behavior = np.array([25.0, 0.5, 0.0, 0.0])
        update = pool.store_candidate(
            Feature("f1"),
            behavior,
            signal,
            fitness=1.25,
        )
        assert update.stored is True
        assert update.replaced is False
        assert pool.size == 1
        best = pool.best(1)
        assert len(best) == 1
        assert best[0][1] == 1.25

    def test_replaces_weaker_incumbent(self):
        pool = DiscoveryPool()
        signal = np.random.randn(100)
        behavior = np.array([25.0, 0.5, 0.0, 0.0])
        pool.store_candidate(Feature("f1"), behavior, signal, fitness=0.4)
        update = pool.store_candidate(Feature("f2"), behavior, signal, fitness=0.9)
        assert update.stored is True
        assert update.replaced is True
        assert pool.size == 1
        best = pool.best(1)
        assert repr(best[0][0]) == "f2"

    def test_rejects_weaker_incumbent(self):
        pool = DiscoveryPool()
        signal = np.random.randn(100)
        behavior = np.array([25.0, 0.5, 0.0, 0.0])
        pool.store_candidate(Feature("f1"), behavior, signal, fitness=0.9)
        update = pool.store_candidate(Feature("f2"), behavior, signal, fitness=0.4)
        assert update.stored is False
        assert update.replaced is False
        assert pool.size == 1

    def test_rejects_bad_signal(self):
        pool = DiscoveryPool()
        signal = np.ones(100)  # constant → fails sanity
        behavior = np.array([25.0, 0.5, 0.0, 0.0])
        update = pool.store_candidate(Feature("f1"), behavior, signal)
        assert update.stored is False
        assert update.replaced is False
        assert pool.size == 0


# ---------------------------------------------------------------------------
# Behavior tests
# ---------------------------------------------------------------------------

class TestBehavior:
    def test_compute_behavior_shape_4d(self):
        signal = np.random.randn(100)
        expr = Feature("f1")
        b = compute_behavior(signal, expr)
        assert b.shape == (4,)

    def test_compute_behavior_with_prices(self):
        rng = np.random.RandomState(42)
        prices = 100 + np.cumsum(rng.randn(200) * 0.5)
        signal = rng.randn(200)
        expr = Feature("f1")
        b = compute_behavior(signal, expr, prices=prices)
        assert b.shape == (4,)
        assert all(np.isfinite(b))

    def test_persistence_constant_signal(self):
        assert _persistence(np.ones(100)) == 0.0

    def test_persistence_autocorrelated_signal(self):
        rng = np.random.RandomState(42)
        signal = np.zeros(200)
        for i in range(1, 200):
            signal[i] = 0.95 * signal[i - 1] + rng.randn() * 0.1
        hl = _persistence(signal)
        assert hl > 5.0

    def test_activity_all_active(self):
        signal = np.random.randn(100) * 10
        act = _activity(signal)
        assert act > 0.5

    def test_activity_mostly_zero(self):
        signal = np.zeros(100)
        signal[10] = 5.0
        signal[50] = -3.0
        act = _activity(signal)
        assert act < 0.2

    def test_price_beta_no_prices(self):
        signal = np.random.randn(100)
        assert _price_beta(signal, None) == 0.0

    def test_price_beta_momentum_signal(self):
        rng = np.random.RandomState(42)
        prices = 100 + np.cumsum(rng.randn(200) * 0.5)
        rets = np.diff(prices) / (np.abs(prices[:-1]) + 1e-12)
        # Signal that mimics returns → positive beta
        signal = np.concatenate([[0.0], rets])
        beta = _price_beta(signal, rets)
        assert beta > 0.3

    def test_vol_sensitivity_no_prices(self):
        signal = np.random.randn(100)
        assert _vol_sensitivity(signal, None) == 0.0

    def test_vol_sensitivity_returns_finite(self):
        rng = np.random.RandomState(42)
        prices = 100 + np.cumsum(rng.randn(200) * 0.5)
        rets = np.diff(prices) / (np.abs(prices[:-1]) + 1e-12)
        signal = rng.randn(200)
        vs = _vol_sensitivity(signal, rets)
        assert np.isfinite(vs)
        assert -1.0 <= vs <= 1.0
