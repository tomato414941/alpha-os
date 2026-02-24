"""Tests for statistical validation modules (DSR, FDR, PBO)."""
import numpy as np
import pytest

from alpha_os.backtest.cost_model import CostModel
from alpha_os.backtest.engine import BacktestEngine
from alpha_os.validation.deflated_sharpe import (
    DSRResult,
    deflated_sharpe_ratio,
    _expected_max_sharpe,
)
from alpha_os.validation.fdr import benjamini_hochberg, FDRResult
from alpha_os.validation.pbo import probability_of_backtest_overfitting, PBOResult


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

class TestExpectedMaxSharpe:
    def test_single_trial(self):
        assert _expected_max_sharpe(1) == 0.0

    def test_increases_with_trials(self):
        e10 = _expected_max_sharpe(10)
        e100 = _expected_max_sharpe(100)
        e1000 = _expected_max_sharpe(1000)
        assert 0 < e10 < e100 < e1000

    def test_reasonable_range(self):
        # 1000 trials should give E[max] around 2-3
        e = _expected_max_sharpe(1000)
        assert 2.0 < e < 4.0


class TestDeflatedSharpe:
    def test_strong_signal(self):
        rng = np.random.RandomState(42)
        # Strong positive Sharpe
        returns = rng.normal(0.001, 0.01, 500)
        result = deflated_sharpe_ratio(returns, n_trials=10)
        assert isinstance(result, DSRResult)
        assert result.observed_sharpe > 0
        assert 0 <= result.p_value <= 1

    def test_random_noise(self):
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0, 0.01, 500)
        result = deflated_sharpe_ratio(returns, n_trials=100)
        # Random noise + many trials -> not significant
        assert result.p_value > 0.05 or result.observed_sharpe < 0.5

    def test_more_trials_harder(self):
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0005, 0.01, 500)
        r10 = deflated_sharpe_ratio(returns, n_trials=10)
        r1000 = deflated_sharpe_ratio(returns, n_trials=1000)
        # More trials = higher bar = lower deflated sharpe
        assert r1000.deflated_sharpe < r10.deflated_sharpe

    def test_short_returns(self):
        result = deflated_sharpe_ratio(np.array([0.01, 0.02]), n_trials=5)
        assert result.p_value == 1.0

    def test_zero_trials(self):
        result = deflated_sharpe_ratio(np.random.randn(100), n_trials=0)
        assert result.p_value == 1.0


# ---------------------------------------------------------------------------
# FDR (Benjamini-Hochberg)
# ---------------------------------------------------------------------------

class TestFDR:
    def test_empty(self):
        result = benjamini_hochberg([])
        assert result.n_tested == 0
        assert result.n_rejected == 0

    def test_all_significant(self):
        # Very low p-values
        pvals = [0.001, 0.002, 0.003, 0.004, 0.005]
        result = benjamini_hochberg(pvals, alpha=0.05)
        assert result.n_rejected == 5
        assert result.rejected_indices == [0, 1, 2, 3, 4]

    def test_none_significant(self):
        pvals = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = benjamini_hochberg(pvals, alpha=0.05)
        assert result.n_rejected == 0

    def test_mixed(self):
        pvals = [0.001, 0.01, 0.05, 0.5, 0.9]
        result = benjamini_hochberg(pvals, alpha=0.05)
        assert 0 < result.n_rejected < 5

    def test_adjusted_pvalues_monotone(self):
        pvals = [0.01, 0.04, 0.05, 0.2, 0.5]
        result = benjamini_hochberg(pvals, alpha=0.10)
        adj = result.adjusted_pvalues
        # Adjusted p-values for sorted originals should be non-decreasing
        sorted_adj = [adj[i] for i in np.argsort(pvals)]
        for i in range(len(sorted_adj) - 1):
            assert sorted_adj[i] <= sorted_adj[i + 1] + 1e-10

    def test_numpy_input(self):
        pvals = np.array([0.01, 0.02, 0.5])
        result = benjamini_hochberg(pvals, alpha=0.05)
        assert isinstance(result, FDRResult)

    def test_stricter_alpha(self):
        pvals = [0.01, 0.03, 0.05]
        r_loose = benjamini_hochberg(pvals, alpha=0.10)
        r_strict = benjamini_hochberg(pvals, alpha=0.01)
        assert r_strict.n_rejected <= r_loose.n_rejected


# ---------------------------------------------------------------------------
# PBO
# ---------------------------------------------------------------------------

class TestPBO:
    @staticmethod
    def _make_data(n_strategies=5, n_days=200, seed=42):
        rng = np.random.RandomState(seed)
        signals = rng.randn(n_strategies, n_days)
        prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_days))
        return signals, prices

    def test_basic(self):
        signals, prices = self._make_data()
        engine = BacktestEngine()
        result = probability_of_backtest_overfitting(
            signals, prices, engine, n_blocks=4, max_combinations=6
        )
        assert isinstance(result, PBOResult)
        assert 0 <= result.pbo <= 1

    def test_random_signals_high_pbo(self):
        # Random signals should have high PBO (no real edge)
        signals, prices = self._make_data(n_strategies=10, n_days=300)
        engine = BacktestEngine()
        result = probability_of_backtest_overfitting(
            signals, prices, engine, n_blocks=4, max_combinations=6
        )
        # PBO should be moderate-to-high for random signals
        assert result.pbo >= 0.0

    def test_too_few_strategies(self):
        signals = np.random.randn(1, 200)
        prices = np.cumprod(1 + np.random.randn(200) * 0.01) * 100
        engine = BacktestEngine()
        result = probability_of_backtest_overfitting(
            signals, prices, engine, n_blocks=4
        )
        assert result.pbo == 1.0

    def test_too_short(self):
        signals = np.random.randn(5, 20)
        prices = np.cumprod(1 + np.random.randn(20) * 0.01) * 100
        engine = BacktestEngine()
        result = probability_of_backtest_overfitting(
            signals, prices, engine, n_blocks=10
        )
        assert result.pbo == 1.0

    def test_distributions_populated(self):
        signals, prices = self._make_data()
        engine = BacktestEngine()
        result = probability_of_backtest_overfitting(
            signals, prices, engine, n_blocks=4, max_combinations=3
        )
        assert len(result.logit_distribution) > 0
        assert len(result.oos_sharpe_distribution) > 0
        assert len(result.logit_distribution) == result.n_combinations
