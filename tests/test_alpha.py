"""Tests for alpha registry, lifecycle, combiner, and governance gates."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from alpha_os.alpha.registry import AlphaRegistry, AlphaRecord, AlphaState
from alpha_os.alpha.lifecycle import AlphaLifecycle, LifecycleConfig
from alpha_os.alpha.combiner import (
    select_low_correlation,
    equal_weight_combine,
    CombinerConfig,
)
from alpha_os.governance.gates import adoption_gate, GateConfig, GateResult


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestAlphaRegistry:
    def _make_registry(self, tmp_path):
        return AlphaRegistry(db_path=tmp_path / "test.db")

    def test_register_and_get(self, tmp_path):
        reg = self._make_registry(tmp_path)
        rec = AlphaRecord(alpha_id="a1", expression="(neg nvda)", fitness=0.5)
        reg.register(rec)
        got = reg.get("a1")
        assert got is not None
        assert got.expression == "(neg nvda)"
        assert got.state == AlphaState.BORN
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
        reg.register(AlphaRecord(alpha_id="a1", expression="x", state=AlphaState.BORN))
        reg.register(AlphaRecord(alpha_id="a2", expression="y", state=AlphaState.ACTIVE))
        reg.register(AlphaRecord(alpha_id="a3", expression="z", state=AlphaState.BORN))
        born = reg.list_by_state(AlphaState.BORN)
        assert len(born) == 2
        active = reg.list_active()
        assert len(active) == 1
        reg.close()

    def test_count(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(AlphaRecord(alpha_id="a1", expression="x"))
        reg.register(AlphaRecord(alpha_id="a2", expression="y"))
        assert reg.count() == 2
        assert reg.count(state=AlphaState.BORN) == 2
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


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestAlphaLifecycle:
    def _setup(self, tmp_path):
        reg = AlphaRegistry(db_path=tmp_path / "test.db")
        cfg = LifecycleConfig(
            oos_sharpe_min=0.5,
            pbo_max=0.5,
            dsr_pvalue_max=0.05,
            correlation_max=0.5,
        )
        lc = AlphaLifecycle(reg, config=cfg)
        return reg, lc

    def test_born_passes_gate(self, tmp_path):
        reg, lc = self._setup(tmp_path)
        reg.register(AlphaRecord(
            alpha_id="a1", expression="x",
            oos_sharpe=1.0, pbo=0.2, dsr_pvalue=0.01, correlation_avg=0.1,
        ))
        state = lc.evaluate_born("a1")
        assert state == AlphaState.ACTIVE

    def test_born_fails_gate(self, tmp_path):
        reg, lc = self._setup(tmp_path)
        reg.register(AlphaRecord(
            alpha_id="a1", expression="x",
            oos_sharpe=0.1, pbo=0.8, dsr_pvalue=0.5,
        ))
        state = lc.evaluate_born("a1")
        assert state == AlphaState.RETIRED

    def test_active_to_probation(self, tmp_path):
        reg, lc = self._setup(tmp_path)
        reg.register(AlphaRecord(
            alpha_id="a1", expression="x", state=AlphaState.ACTIVE,
            oos_sharpe=1.0, pbo=0.2, dsr_pvalue=0.01,
        ))
        state = lc.evaluate_active("a1", live_sharpe=0.1)
        assert state == AlphaState.PROBATION

    def test_active_stays(self, tmp_path):
        reg, lc = self._setup(tmp_path)
        reg.register(AlphaRecord(
            alpha_id="a1", expression="x", state=AlphaState.ACTIVE,
        ))
        state = lc.evaluate_active("a1", live_sharpe=0.8)
        assert state == AlphaState.ACTIVE

    def test_probation_to_retired(self, tmp_path):
        reg, lc = self._setup(tmp_path)
        reg.register(AlphaRecord(
            alpha_id="a1", expression="x", state=AlphaState.PROBATION,
        ))
        state = lc.evaluate_probation("a1", live_sharpe=-0.5)
        assert state == AlphaState.RETIRED

    def test_probation_to_active(self, tmp_path):
        reg, lc = self._setup(tmp_path)
        reg.register(AlphaRecord(
            alpha_id="a1", expression="x", state=AlphaState.PROBATION,
        ))
        state = lc.evaluate_probation("a1", live_sharpe=0.8)
        assert state == AlphaState.ACTIVE

    def test_not_found_raises(self, tmp_path):
        reg, lc = self._setup(tmp_path)
        with pytest.raises(ValueError):
            lc.evaluate_born("nonexistent")


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

    def test_equal_weight_combine(self):
        signals = np.array([
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
        ])
        combined = equal_weight_combine(signals, [0, 1])
        assert combined.shape == (3,)
        np.testing.assert_array_equal(combined, [0.0, 0.0, 0.0])

    def test_equal_weight_empty(self):
        signals = np.random.randn(5, 100)
        combined = equal_weight_combine(signals, [])
        assert np.all(combined == 0.0)


# ---------------------------------------------------------------------------
# Governance Gates
# ---------------------------------------------------------------------------

class TestGovernanceGates:
    def test_all_pass(self):
        result = adoption_gate(
            oos_sharpe=1.0, pbo=0.2, dsr_pvalue=0.01,
            fdr_passed=True, avg_correlation=0.1, n_days=500,
        )
        assert result.passed is True
        assert all(result.checks.values())
        assert len(result.reasons) == 0

    def test_sharpe_fail(self):
        result = adoption_gate(
            oos_sharpe=0.1, pbo=0.2, dsr_pvalue=0.01,
            fdr_passed=True, avg_correlation=0.1, n_days=500,
        )
        assert result.passed is False
        assert not result.checks["oos_sharpe"]
        assert any("OOS Sharpe" in r for r in result.reasons)

    def test_pbo_fail(self):
        result = adoption_gate(
            oos_sharpe=1.0, pbo=0.8, dsr_pvalue=0.01,
            fdr_passed=True, avg_correlation=0.1, n_days=500,
        )
        assert result.passed is False
        assert not result.checks["pbo"]

    def test_multiple_failures(self):
        result = adoption_gate(
            oos_sharpe=0.1, pbo=0.8, dsr_pvalue=0.5,
            fdr_passed=False, avg_correlation=0.9, n_days=50,
        )
        assert result.passed is False
        assert len(result.reasons) == 6

    def test_custom_config(self):
        cfg = GateConfig(oos_sharpe_min=0.3, pbo_max=0.8, dsr_pvalue_max=0.10)
        result = adoption_gate(
            oos_sharpe=0.4, pbo=0.6, dsr_pvalue=0.08,
            fdr_passed=True, avg_correlation=0.1, n_days=500,
            config=cfg,
        )
        assert result.passed is True

    def test_fdr_not_required(self):
        cfg = GateConfig(fdr_pass_required=False)
        result = adoption_gate(
            oos_sharpe=1.0, pbo=0.2, dsr_pvalue=0.01,
            fdr_passed=False, avg_correlation=0.1, n_days=500,
            config=cfg,
        )
        assert result.checks["fdr"] is True
