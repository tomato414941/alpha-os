"""Tests for backtest engine and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_os.backtest.cost_model import CostModel
from alpha_os.backtest.engine import BacktestEngine, BacktestResult
from alpha_os.backtest import metrics


class TestCostModel:
    def test_one_way_cost(self):
        cm = CostModel(commission_pct=0.10, slippage_pct=0.05)
        assert cm.one_way_cost == pytest.approx(0.0015)

    def test_zero_cost(self):
        cm = CostModel(commission_pct=0.0, slippage_pct=0.0)
        assert cm.one_way_cost == 0.0

    def test_round_trip(self):
        cm = CostModel(commission_pct=0.10, slippage_pct=0.05)
        assert cm.round_trip_cost(100.0) == pytest.approx(0.15)


class TestMetrics:
    def test_sharpe_positive(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 252)
        s = metrics.sharpe_ratio(returns)
        assert s > 0

    def test_sharpe_zero_std(self):
        returns = np.zeros(100)
        assert metrics.sharpe_ratio(returns) == 0.0

    def test_max_drawdown(self):
        # Price goes 100 -> 80 -> 90, dd = 20%
        returns = np.array([0.0, -0.10, -0.10, 0.05, 0.05])
        dd = metrics.max_drawdown(returns)
        assert dd > 0.15

    def test_max_drawdown_no_loss(self):
        returns = np.array([0.01, 0.02, 0.01])
        assert metrics.max_drawdown(returns) == pytest.approx(0.0)

    def test_turnover(self):
        positions = np.array([0.0, 1.0, 1.0, -1.0, 0.0])
        t = metrics.turnover(positions)
        # changes: 1.0, 0.0, 2.0, 1.0 -> mean = 1.0
        assert t == pytest.approx(1.0)

    def test_annual_return(self):
        # 1% daily for 252 days
        returns = np.full(252, 0.01)
        ar = metrics.annual_return(returns)
        assert ar > 1.0  # > 100% annualized

    def test_sortino_positive(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 252)
        s = metrics.sortino_ratio(returns)
        assert s > 0

    def test_cvar_negative_on_loss_tail(self):
        returns = np.array([-0.10, -0.05, -0.02, 0.01, 0.02])
        assert metrics.cvar(returns, alpha=0.4) < 0.0

    def test_expected_log_growth_positive_for_positive_returns(self):
        returns = np.full(252, 0.001)
        assert metrics.expected_log_growth(returns) > 0.0

    def test_tail_hit_rate_in_range(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, 252)
        thr = metrics.tail_hit_rate(returns, sigma=2.0)
        assert 0.0 <= thr <= 1.0


class TestBacktestEngine:
    def test_run_basic(self):
        prices = np.linspace(100, 120, 252)
        alpha = np.ones(252)  # always long
        engine = BacktestEngine(CostModel(commission_pct=0.0, slippage_pct=0.0))
        result = engine.run(alpha, prices, alpha_id="test")
        assert isinstance(result, BacktestResult)
        assert result.n_days == 251  # 252 prices -> 251 returns
        assert result.annual_return > 0  # trending up, always long
        assert 0.0 <= result.tail_hit_rate <= 1.0

    def test_run_zero_signal(self):
        prices = np.linspace(100, 120, 100)
        alpha = np.zeros(100)
        engine = BacktestEngine()
        result = engine.run(alpha, prices)
        assert result.sharpe == 0.0
        assert result.expected_log_growth == 0.0

    def test_costs_reduce_returns(self):
        prices = np.linspace(100, 110, 100)
        rng = np.random.default_rng(42)
        alpha = rng.standard_normal(100)  # noisy signal = high turnover
        no_cost = BacktestEngine(CostModel(0.0, 0.0))
        with_cost = BacktestEngine(CostModel(0.5, 0.5))
        r_free = no_cost.run(alpha, prices)
        r_costly = with_cost.run(alpha, prices)
        assert r_costly.annual_return < r_free.annual_return

    def test_run_batch(self):
        prices = np.linspace(100, 120, 100)
        rng = np.random.default_rng(42)
        signals = rng.standard_normal((50, 100))
        engine = BacktestEngine()
        results = engine.run_batch(signals, prices)
        assert len(results) == 50
        assert all(isinstance(r, BacktestResult) for r in results)
