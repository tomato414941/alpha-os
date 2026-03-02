"""Tests for paper trading — tracker persistence and trader orchestration."""
from __future__ import annotations

import pytest

from alpha_os.execution.executor import Fill
from alpha_os.paper.tracker import PaperPortfolioTracker, PortfolioSnapshot


class TestPaperPortfolioTracker:
    def test_save_and_load_snapshot(self, tmp_path):
        db = tmp_path / "test.db"
        tracker = PaperPortfolioTracker(db)
        snap = PortfolioSnapshot(
            date="2026-01-01",
            cash=9000.0,
            positions={"btc_ohlcv": 0.01},
            portfolio_value=10000.0,
            daily_pnl=50.0,
            daily_return=0.005,
            combined_signal=0.5,
            dd_scale=1.0,
            vol_scale=0.9,
        )
        tracker.save_snapshot(snap)

        loaded = tracker.get_last_snapshot()
        assert loaded is not None
        assert loaded.date == "2026-01-01"
        assert loaded.cash == 9000.0
        assert loaded.positions == {"btc_ohlcv": 0.01}
        assert loaded.portfolio_value == 10000.0

        by_date = tracker.get_snapshot("2026-01-01")
        assert by_date is not None
        assert by_date.daily_pnl == 50.0
        tracker.close()

    def test_get_returns(self, tmp_path):
        db = tmp_path / "test.db"
        tracker = PaperPortfolioTracker(db)
        for i, ret in enumerate([0.01, -0.005, 0.02]):
            tracker.save_snapshot(PortfolioSnapshot(
                date=f"2026-01-0{i+1}",
                cash=10000.0, positions={},
                portfolio_value=10000.0 * (1 + ret),
                daily_pnl=10000.0 * ret,
                daily_return=ret,
                combined_signal=0.0, dd_scale=1.0, vol_scale=1.0,
            ))
        returns = tracker.get_returns()
        assert len(returns) == 3
        assert returns[0] == pytest.approx(0.01)
        assert returns[1] == pytest.approx(-0.005)
        tracker.close()

    def test_equity_curve(self, tmp_path):
        db = tmp_path / "test.db"
        tracker = PaperPortfolioTracker(db)
        tracker.save_snapshot(PortfolioSnapshot(
            date="2026-01-01", cash=10000.0, positions={},
            portfolio_value=10000.0, daily_pnl=0.0, daily_return=0.0,
            combined_signal=0.0, dd_scale=1.0, vol_scale=1.0,
        ))
        tracker.save_snapshot(PortfolioSnapshot(
            date="2026-01-02", cash=10050.0, positions={},
            portfolio_value=10050.0, daily_pnl=50.0, daily_return=0.005,
            combined_signal=0.0, dd_scale=1.0, vol_scale=1.0,
        ))
        curve = tracker.get_equity_curve()
        assert len(curve) == 2
        assert curve[0] == ("2026-01-01", 10000.0)
        assert curve[1] == ("2026-01-02", 10050.0)
        tracker.close()

    def test_save_fills(self, tmp_path):
        db = tmp_path / "test.db"
        tracker = PaperPortfolioTracker(db)
        fills = [Fill(symbol="btc_ohlcv", side="buy", qty=0.01, price=97000.0, order_id="p-1")]
        tracker.save_fills("2026-01-01", fills)
        assert tracker.get_total_trades() == 1
        tracker.close()

    def test_summary(self, tmp_path):
        db = tmp_path / "test.db"
        tracker = PaperPortfolioTracker(db)
        base = 10000.0
        daily_rets = [0.01, 0.005, 0.02, -0.003, 0.015]
        val = base
        for i, ret in enumerate(daily_rets):
            pnl = val * ret
            val += pnl
            tracker.save_snapshot(PortfolioSnapshot(
                date=f"2026-01-{i+1:02d}",
                cash=val / 2, positions={"btc_ohlcv": 0.001},
                portfolio_value=val,
                daily_pnl=pnl,
                daily_return=ret,
                combined_signal=0.3, dd_scale=1.0, vol_scale=1.0,
            ))
        summary = tracker.summary()
        assert summary is not None
        assert summary.n_days == 5
        assert summary.total_return > 0
        assert summary.sharpe > 0
        assert summary.max_drawdown >= 0
        tracker.close()

    def test_empty_tracker(self, tmp_path):
        db = tmp_path / "test.db"
        tracker = PaperPortfolioTracker(db)
        assert tracker.get_last_snapshot() is None
        assert tracker.get_returns() == []
        assert tracker.summary() is None
        tracker.close()

    def test_alpha_signals(self, tmp_path):
        db = tmp_path / "test.db"
        tracker = PaperPortfolioTracker(db)
        tracker.save_alpha_signals("2026-01-01", {"a1": 0.5, "a2": -0.3})
        tracker.save_alpha_signals("2026-01-01", {"a1": 0.6})  # upsert
        tracker.close()


class TestPositionSizing:
    def test_dollar_pos_scales_with_portfolio_value(self, tmp_path):
        """Position sizing should use current portfolio value, not initial capital."""
        from alpha_os.paper.trader import PaperTrader
        from alpha_os.config import Config
        from alpha_os.alpha.registry import AlphaRegistry
        from alpha_os.forward.tracker import ForwardTracker
        from alpha_os.data.store import DataStore
        from alpha_os.execution.paper import PaperExecutor
        from alpha_os.governance.audit_log import AuditLog

        cfg = Config()
        pt = PaperPortfolioTracker(tmp_path / "paper.db")

        # Simulate a grown portfolio: started at $10k, now $50k
        executor = PaperExecutor(initial_cash=50000.0)
        trader = PaperTrader(
            asset="BTC",
            config=cfg,
            portfolio_tracker=pt,
            registry=AlphaRegistry(tmp_path / "reg.db"),
            forward_tracker=ForwardTracker(tmp_path / "fwd.db"),
            executor=executor,
            audit_log=AuditLog(tmp_path / "audit.jsonl"),
            store=DataStore(tmp_path / "cache.db"),
        )

        # Verify that position sizing uses prev_value (portfolio_value) not initial_capital
        adjusted = 0.5
        prev_value = trader.executor.portfolio_value  # $50,000
        max_pos = trader.max_position_pct  # 1.0

        expected_dollar_pos = adjusted * prev_value * max_pos  # 0.5 * 50000 * 1.0 = 25000
        wrong_dollar_pos = adjusted * trader.initial_capital * max_pos  # 0.5 * 10000 * 1.0 = 5000

        assert expected_dollar_pos == 25000.0
        assert wrong_dollar_pos == 5000.0
        assert expected_dollar_pos != wrong_dollar_pos

        trader.close()


class TestPaperTrader:
    def test_restore_state_empty(self, tmp_path):
        """Fresh trader should have initial capital."""
        from alpha_os.paper.trader import PaperTrader
        from alpha_os.config import Config
        from alpha_os.alpha.registry import AlphaRegistry
        from alpha_os.forward.tracker import ForwardTracker
        from alpha_os.data.store import DataStore
        from alpha_os.execution.paper import PaperExecutor
        from alpha_os.governance.audit_log import AuditLog

        db = tmp_path / "paper.db"
        cfg = Config()
        pt = PaperPortfolioTracker(db)

        trader = PaperTrader(
            asset="BTC",
            config=cfg,
            portfolio_tracker=pt,
            registry=AlphaRegistry(tmp_path / "reg.db"),
            forward_tracker=ForwardTracker(tmp_path / "fwd.db"),
            executor=PaperExecutor(initial_cash=cfg.trading.initial_capital),
            audit_log=AuditLog(tmp_path / "audit.jsonl"),
            store=DataStore(tmp_path / "cache.db"),
        )
        assert trader.executor.get_cash() == pytest.approx(cfg.trading.initial_capital)
        trader.close()

    def test_restore_state_from_snapshot(self, tmp_path):
        """Trader should restore cash/positions from last snapshot."""
        from alpha_os.paper.trader import PaperTrader
        from alpha_os.config import Config
        from alpha_os.alpha.registry import AlphaRegistry
        from alpha_os.forward.tracker import ForwardTracker
        from alpha_os.data.store import DataStore
        from alpha_os.execution.paper import PaperExecutor
        from alpha_os.governance.audit_log import AuditLog

        db = tmp_path / "paper.db"
        pt = PaperPortfolioTracker(db)
        pt.save_snapshot(PortfolioSnapshot(
            date="2026-01-05", cash=8500.0,
            positions={"btc_ohlcv": 0.015},
            portfolio_value=9950.0, daily_pnl=-50.0, daily_return=-0.005,
            combined_signal=0.3, dd_scale=1.0, vol_scale=0.9,
        ))

        cfg = Config()
        trader = PaperTrader(
            asset="BTC",
            config=cfg,
            portfolio_tracker=pt,
            registry=AlphaRegistry(tmp_path / "reg.db"),
            forward_tracker=ForwardTracker(tmp_path / "fwd.db"),
            executor=PaperExecutor(initial_cash=cfg.trading.initial_capital),
            audit_log=AuditLog(tmp_path / "audit.jsonl"),
            store=DataStore(tmp_path / "cache.db"),
        )
        assert trader.executor.get_cash() == pytest.approx(8500.0)
        assert trader.executor.get_position("btc_ohlcv") == pytest.approx(0.015)
        trader.close()
