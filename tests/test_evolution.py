"""Tests for the evolution module (GP, archive, behavior)."""
import numpy as np
import pytest

from alpha_os.dsl.expr import Feature, UnaryOp, BinaryOp, RollingOp, Constant, ConditionalOp, LagOp
from alpha_os.dsl.generator import AlphaGenerator
from alpha_os.evolution.gp import GPConfig, GPEvolver, _tree_depth, _node_count, crossover
from alpha_os.evolution.archive import AlphaArchive, ArchiveConfig
from alpha_os.evolution.behavior import (
    compute_behavior,
    _avg_abs_corr,
    _holding_half_life,
    _signal_turnover,
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


class TestCrossover:
    def test_crossover_returns_two_exprs(self):
        import random

        rng = random.Random(42)
        p1 = BinaryOp("add", Feature("f1"), Feature("f2"))
        p2 = BinaryOp("sub", Feature("f3"), Feature("f1"))
        c1, c2 = crossover(p1, p2, rng)
        assert isinstance(c1, BinaryOp)
        assert isinstance(c2, BinaryOp)

    def test_crossover_small_trees(self):
        import random

        rng = random.Random(42)
        p1 = Feature("f1")
        p2 = Feature("f2")
        c1, c2 = crossover(p1, p2, rng)
        assert repr(c1) == "f1"
        assert repr(c2) == "f2"


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


# ---------------------------------------------------------------------------
# Archive tests
# ---------------------------------------------------------------------------

class TestAlphaArchive:
    def test_add_new_cell(self):
        archive = AlphaArchive()
        expr = Feature("f1")
        behavior = np.array([0.5, 50.0, 1.0, 5.0])
        assert archive.add(expr, 0.5, behavior) is True
        assert archive.size == 1

    def test_add_better_replaces(self):
        archive = AlphaArchive()
        behavior = np.array([0.5, 50.0, 1.0, 5.0])
        archive.add(Feature("f1"), 0.3, behavior)
        assert archive.add(Feature("f2"), 0.7, behavior) is True
        assert archive.size == 1
        best = archive.best(1)
        assert repr(best[0][0]) == "f2"

    def test_add_worse_rejected(self):
        archive = AlphaArchive()
        behavior = np.array([0.5, 50.0, 1.0, 5.0])
        archive.add(Feature("f1"), 0.7, behavior)
        assert archive.add(Feature("f2"), 0.3, behavior) is False
        assert archive.size == 1

    def test_different_cells(self):
        archive = AlphaArchive()
        archive.add(Feature("f1"), 0.5, np.array([0.1, 10.0, 0.5, 3.0]))
        archive.add(Feature("f2"), 0.6, np.array([0.9, 90.0, 1.5, 15.0]))
        assert archive.size == 2

    def test_capacity(self):
        cfg = ArchiveConfig(dims=(5, 5, 5, 5))
        archive = AlphaArchive(config=cfg)
        assert archive.capacity == 625

    def test_coverage(self):
        cfg = ArchiveConfig(dims=(2, 2, 2, 2))
        archive = AlphaArchive(config=cfg)
        archive.add(Feature("f1"), 0.5, np.array([0.2, 20.0, 0.5, 5.0]))
        assert archive.coverage == 1 / 16

    def test_best_ordering(self):
        archive = AlphaArchive()
        archive.add(Feature("f1"), 0.3, np.array([0.1, 10.0, 0.5, 3.0]))
        archive.add(Feature("f2"), 0.9, np.array([0.5, 50.0, 1.0, 10.0]))
        archive.add(Feature("f3"), 0.6, np.array([0.9, 90.0, 1.5, 15.0]))
        best = archive.best(3)
        assert best[0][1] >= best[1][1] >= best[2][1]

    def test_sample(self):
        archive = AlphaArchive()
        for i in range(5):
            archive.add(
                Feature(f"f{i}"),
                float(i),
                np.array([i * 0.1, i * 10.0, i * 0.3, i + 1.0]),
            )
        import random

        sampled = archive.sample(3, rng=random.Random(42))
        assert len(sampled) == 3

    def test_elites(self):
        archive = AlphaArchive()
        archive.add(Feature("f1"), 0.5, np.array([0.5, 50.0, 1.0, 5.0]))
        elites = archive.elites()
        assert len(elites) == 1
        assert elites[0][1] == 0.5

    def test_boundary_clipping(self):
        archive = AlphaArchive()
        behavior = np.array([2.0, 200.0, 5.0, 50.0])  # all out of range
        assert archive.add(Feature("f1"), 0.5, behavior) is True


# ---------------------------------------------------------------------------
# Behavior tests
# ---------------------------------------------------------------------------

class TestBehavior:
    def test_compute_behavior_shape(self):
        signal = np.random.randn(100)
        expr = Feature("f1")
        b = compute_behavior(signal, expr)
        assert b.shape == (4,)

    def test_corr_no_live_signals(self):
        signal = np.random.randn(100)
        assert _avg_abs_corr(signal, None) == 0.0
        assert _avg_abs_corr(signal, []) == 0.0

    def test_corr_with_identical(self):
        signal = np.random.randn(100)
        corr = _avg_abs_corr(signal, [signal])
        assert corr == pytest.approx(1.0, abs=0.01)

    def test_corr_with_uncorrelated(self):
        rng = np.random.RandomState(42)
        s1 = rng.randn(1000)
        s2 = rng.randn(1000)
        corr = _avg_abs_corr(s1, [s2])
        assert corr < 0.1

    def test_holding_half_life_constant(self):
        signal = np.ones(100)
        assert _holding_half_life(signal) == 0.0

    def test_holding_half_life_positive(self):
        # AR(1) with high autocorrelation
        rng = np.random.RandomState(42)
        signal = np.zeros(200)
        for i in range(1, 200):
            signal[i] = 0.95 * signal[i - 1] + rng.randn() * 0.1
        hl = _holding_half_life(signal)
        assert hl > 5.0  # high autocorrelation -> long half-life

    def test_turnover_constant(self):
        signal = np.ones(100)
        assert _signal_turnover(signal) == 0.0

    def test_turnover_positive(self):
        signal = np.random.randn(100)
        t = _signal_turnover(signal)
        assert t > 0.0

    def test_complexity_is_node_count(self):
        expr = BinaryOp("add", Feature("f1"), Feature("f2"))
        signal = np.random.randn(100)
        b = compute_behavior(signal, expr)
        assert b[3] == 3.0  # binary + 2 leaves
