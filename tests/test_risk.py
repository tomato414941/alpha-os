"""Tests for risk management and position sizing."""
import numpy as np
import pytest

from alpha_os.risk.manager import RiskManager, RiskConfig
from alpha_os.risk.position import (
    PositionConfig,
    signal_to_positions,
    compute_pnl,
)


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class TestRiskManager:
    def test_no_drawdown(self):
        rm = RiskManager()
        rm.reset(1.0)
        rm.update_equity(1.05)
        assert rm.current_drawdown == 0.0
        assert rm.dd_scale == 1.0

    def test_stage1_drawdown(self):
        cfg = RiskConfig(dd_stage1_pct=0.05, dd_stage1_scale=0.75)
        rm = RiskManager(config=cfg)
        rm.reset(1.0)
        rm.update_equity(1.0)
        rm.update_equity(0.93)  # 7% DD
        assert rm.current_drawdown == pytest.approx(0.07, abs=0.01)
        assert rm.dd_scale == 0.75

    def test_stage2_drawdown(self):
        cfg = RiskConfig(dd_stage2_pct=0.10, dd_stage2_scale=0.50)
        rm = RiskManager(config=cfg)
        rm.reset(1.0)
        rm.update_equity(0.88)  # 12% DD
        assert rm.dd_scale == 0.50

    def test_stage3_drawdown(self):
        cfg = RiskConfig(dd_stage3_pct=0.15, dd_stage3_scale=0.25)
        rm = RiskManager(config=cfg)
        rm.reset(1.0)
        rm.update_equity(0.80)  # 20% DD
        assert rm.dd_scale == 0.25

    def test_vol_scale_high_vol(self):
        rm = RiskManager(config=RiskConfig(target_vol=0.15))
        # High realized vol -> scale down
        returns = np.random.RandomState(42).normal(0, 0.03, 63)
        scale = rm.vol_scale(returns)
        assert scale < 1.0

    def test_vol_scale_low_vol(self):
        rm = RiskManager(config=RiskConfig(target_vol=0.15))
        returns = np.random.RandomState(42).normal(0, 0.003, 63)
        scale = rm.vol_scale(returns)
        assert scale > 1.0

    def test_vol_scale_capped(self):
        rm = RiskManager(config=RiskConfig(target_vol=0.15))
        # Near-zero vol should cap at 2.0
        returns = np.full(63, 0.0001)
        scale = rm.vol_scale(returns)
        assert scale <= 2.0

    def test_adjust_position(self):
        rm = RiskManager()
        rm.reset(1.0)
        rm.update_equity(0.93)  # stage 1
        returns = np.random.RandomState(42).normal(0, 0.01, 63)
        pos = rm.adjust_position(0.8, returns)
        assert abs(pos) < 0.8  # should be scaled down

    def test_adjust_positions_vectorized(self):
        rm = RiskManager()
        n = 200
        rng = np.random.RandomState(42)
        raw_pos = rng.randn(n) * 0.5
        returns = rng.normal(0.0005, 0.01, n)
        adjusted = rm.adjust_positions(raw_pos, returns)
        assert adjusted.shape == (n,)
        assert np.all(np.abs(adjusted) <= 1.0)

    def test_adjust_positions_reduces_in_drawdown(self):
        cfg = RiskConfig(
            dd_stage1_pct=0.03, dd_stage1_scale=0.5,
            target_vol=100.0,  # effectively disable vol scaling
        )
        rm = RiskManager(config=cfg)
        n = 100
        # Constant positive position, but with losses
        raw_pos = np.ones(n) * 0.5
        returns = np.full(n, -0.005)  # steady losses -> drawdown builds
        adjusted = rm.adjust_positions(raw_pos, returns)
        # Later positions should be smaller due to DD response
        assert adjusted[-1] < adjusted[0]


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------

class TestPositionSizing:
    def test_basic(self):
        signal = np.array([1.0, -1.0, 0.5, 0.0])
        prices = np.array([100.0, 100.0, 100.0, 100.0])
        cfg = PositionConfig(capital=10000.0)
        shares = signal_to_positions(signal, prices, config=cfg)
        assert shares[0] == pytest.approx(100.0)  # $10k / $100
        assert shares[1] == pytest.approx(-100.0)
        assert shares[2] == pytest.approx(50.0)
        assert shares[3] == 0.0

    def test_min_trade_filter(self):
        signal = np.array([0.001, 1.0])
        prices = np.array([100.0, 100.0])
        cfg = PositionConfig(capital=10000.0, min_trade_usd=50.0)
        shares = signal_to_positions(signal, prices, config=cfg)
        assert shares[0] == 0.0  # $10 < $50 minimum
        assert shares[1] != 0.0

    def test_zero_price(self):
        signal = np.array([1.0])
        prices = np.array([0.0])
        shares = signal_to_positions(signal, prices)
        assert shares[0] == 0.0


class TestComputePnl:
    def test_basic(self):
        shares = np.array([10.0, 10.0, 10.0])
        prices = np.array([100.0, 102.0, 101.0])
        pnl = compute_pnl(shares, prices)
        np.testing.assert_array_almost_equal(pnl, [20.0, -10.0])

    def test_short(self):
        shares = np.array([-5.0, -5.0])
        prices = np.array([100.0, 98.0])
        pnl = compute_pnl(shares, prices)
        assert pnl[0] == pytest.approx(10.0)

    def test_empty(self):
        pnl = compute_pnl(np.array([1.0]), np.array([100.0]))
        assert len(pnl) == 0
