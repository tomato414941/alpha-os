"""Tests for pipeline runner — integration test."""
import numpy as np
import pytest

from alpha_os.evolution.gp import GPConfig
from alpha_os.governance.gates import GateConfig
from alpha_os.pipeline.runner import PipelineConfig, PipelineRunner, PipelineResult


class TestPipelineRunner:
    @staticmethod
    def _make_data(n_days=300, seed=42):
        rng = np.random.default_rng(seed)
        features = ["f1", "f2", "f3"]
        data = {}
        for feat in features:
            drift = rng.uniform(-0.0005, 0.001)
            vol = rng.uniform(0.005, 0.03)
            returns = rng.normal(drift, vol, n_days)
            data[feat] = 100.0 * np.cumprod(1.0 + returns)
        prices = data["f1"]
        return features, data, prices

    def test_run_basic(self):
        features, data, prices = self._make_data()
        cfg = PipelineConfig(
            gp=GPConfig(pop_size=20, n_generations=3, max_depth=2),
            gate=GateConfig(
                oos_sharpe_min=0.0,  # relaxed for test
                pbo_max=1.0,
                dsr_pvalue_max=1.0,
                fdr_pass_required=False,
                min_n_days=10,
            ),
        )
        runner = PipelineRunner(features, data, prices, config=cfg, seed=42)
        result = runner.run()
        assert isinstance(result, PipelineResult)
        assert result.n_generated > 0
        assert result.elapsed > 0

    def test_run_returns_combined_signal(self):
        features, data, prices = self._make_data()
        cfg = PipelineConfig(
            gp=GPConfig(pop_size=20, n_generations=3, max_depth=2),
            gate=GateConfig(
                oos_sharpe_min=0.0,
                pbo_max=1.0,
                dsr_pvalue_max=1.0,
                fdr_pass_required=False,
                min_n_days=10,
            ),
        )
        runner = PipelineRunner(features, data, prices, config=cfg, seed=42)
        result = runner.run()
        if result.n_adopted > 0:
            assert result.combined_signal is not None
            assert len(result.combined_signal) == len(prices)

    def test_strict_gates_reject_all(self):
        features, data, prices = self._make_data()
        cfg = PipelineConfig(
            gp=GPConfig(pop_size=10, n_generations=2),
            gate=GateConfig(
                oos_sharpe_min=10.0,  # impossible
                dsr_pvalue_max=0.001,
            ),
        )
        runner = PipelineRunner(features, data, prices, config=cfg, seed=42)
        result = runner.run()
        assert result.n_adopted == 0
        assert result.combined_signal is None

    def test_archive_populated(self):
        features, data, prices = self._make_data()
        cfg = PipelineConfig(
            gp=GPConfig(pop_size=30, n_generations=3),
            gate=GateConfig(
                oos_sharpe_min=0.0,
                pbo_max=1.0,
                dsr_pvalue_max=1.0,
                fdr_pass_required=False,
                min_n_days=10,
            ),
        )
        runner = PipelineRunner(features, data, prices, config=cfg, seed=42)
        runner.run()
        assert runner.archive.size > 0
