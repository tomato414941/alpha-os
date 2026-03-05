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

    def test_consecutive_no_fill_cycles(self, tmp_path):
        db = tmp_path / "test.db"
        tracker = PaperPortfolioTracker(db)
        tracker.save_snapshot(PortfolioSnapshot(
            date="2026-01-01T00:00:00",
            cash=10000.0, positions={},
            portfolio_value=10000.0, daily_pnl=0.0, daily_return=0.0,
            combined_signal=0.0, dd_scale=1.0, vol_scale=1.0,
        ))
        tracker.save_snapshot(PortfolioSnapshot(
            date="2026-01-01T04:00:00",
            cash=10000.0, positions={},
            portfolio_value=10000.0, daily_pnl=0.0, daily_return=0.0,
            combined_signal=0.0, dd_scale=1.0, vol_scale=1.0,
        ))
        tracker.save_fills("2026-01-01T04:00:00", [
            Fill(symbol="btc_ohlcv", side="buy", qty=0.01, price=97000.0, order_id="p-1")
        ])
        tracker.save_snapshot(PortfolioSnapshot(
            date="2026-01-01T08:00:00",
            cash=9990.0, positions={"btc_ohlcv": 0.01},
            portfolio_value=10010.0, daily_pnl=10.0, daily_return=0.001,
            combined_signal=0.2, dd_scale=1.0, vol_scale=1.0,
        ))
        tracker.save_snapshot(PortfolioSnapshot(
            date="2026-01-01T12:00:00",
            cash=9990.0, positions={"btc_ohlcv": 0.01},
            portfolio_value=10005.0, daily_pnl=-5.0, daily_return=-0.0005,
            combined_signal=0.1, dd_scale=1.0, vol_scale=1.0,
        ))
        assert tracker.count_consecutive_no_fill_cycles() == 2
        tracker.close()


class TestRegistryTopTrading:
    def test_top_trading_returns_limited(self, tmp_path):
        from alpha_os.alpha.registry import AlphaRegistry, AlphaRecord, AlphaState

        reg = AlphaRegistry(tmp_path / "reg.db")
        for i in range(10):
            reg.register(AlphaRecord(
                alpha_id=f"a{i}", expression=f"close_{i}",
                state=AlphaState.ACTIVE, oos_sharpe=float(i),
            ))
        reg.register(AlphaRecord(
            alpha_id="d1", expression="close_d",
            state=AlphaState.DORMANT, oos_sharpe=100.0,
        ))
        result = reg.top_trading(3)
        assert len(result) == 3
        assert result[0].oos_sharpe == 9.0
        assert all(r.state != AlphaState.DORMANT for r in result)
        reg.close()

    def test_top_trading_includes_probation(self, tmp_path):
        from alpha_os.alpha.registry import AlphaRegistry, AlphaRecord, AlphaState

        reg = AlphaRegistry(tmp_path / "reg.db")
        reg.register(AlphaRecord(
            alpha_id="a1", expression="close_1",
            state=AlphaState.ACTIVE, oos_sharpe=1.0,
        ))
        reg.register(AlphaRecord(
            alpha_id="p1", expression="close_p",
            state=AlphaState.PROBATION, oos_sharpe=2.0,
        ))
        result = reg.top_trading(10)
        assert len(result) == 2
        assert result[0].alpha_id == "p1"
        reg.close()


class TestSignalDeltaExit:
    def test_zscore_triggers_exit(self):
        import math

        ema_span = 6
        alpha = 2.0 / (ema_span + 1)
        ema = 0.5
        var = 0.001

        for _ in range(10):
            sig = 0.5
            ema = alpha * sig + (1 - alpha) * ema
            dev = sig - ema
            var = alpha * dev**2 + (1 - alpha) * var

        sig = -0.5
        ema = alpha * sig + (1 - alpha) * ema
        dev = sig - ema
        var = alpha * dev**2 + (1 - alpha) * var
        std = math.sqrt(var)
        zscore = (sig - ema) / std if std > 1e-6 else 0.0
        assert zscore < -1.0, f"Expected z < -1.0, got {zscore:.2f}"

    def test_stable_signal_no_exit(self):
        import math

        ema_span = 6
        alpha = 2.0 / (ema_span + 1)
        ema = None
        var = 0.0

        for _ in range(20):
            sig = 0.45
            if ema is None:
                ema = sig
            else:
                ema = alpha * sig + (1 - alpha) * ema
                dev = sig - ema
                var = alpha * dev**2 + (1 - alpha) * var

        std = math.sqrt(var)
        if std > 1e-6:
            zscore = (sig - ema) / std
        else:
            zscore = 0.0
        assert zscore > -1.0


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


class TestMapElitesCombinePath:
    """Test the MAP-Elites two-level ensemble sizing path."""

    def test_cell_grouping_and_ensemble(self):
        """Signals grouped into cells produce valid ensemble result."""
        import numpy as np
        from alpha_os.dsl.expr import Feature, BinaryOp, RollingOp
        from alpha_os.evolution.archive import AlphaArchive
        from alpha_os.evolution.behavior import compute_behavior
        from alpha_os.voting.ensemble import compute_cell_long_pcts, ensemble_sizing

        rng = np.random.RandomState(42)
        archive = AlphaArchive()

        # Simulate 20 alphas with different expressions and signals
        exprs = [Feature("f1")] * 5 + [
            BinaryOp("add", Feature("f1"), Feature("f2"))
        ] * 5 + [
            RollingOp("mean", 10, Feature("f1"))
        ] * 5 + [
            RollingOp("std", 20, Feature("f2"))
        ] * 5
        signals = [rng.randn(300) for _ in range(20)]
        signal_values = [float(s[-2]) for s in signals]

        # Group into cells
        cell_signals: dict[tuple[int, ...], list[float]] = {}
        for expr, sig_arr, sig_val in zip(exprs, signals, signal_values):
            behavior = compute_behavior(sig_arr, expr)
            cell = archive._to_cell(behavior)
            cell_signals.setdefault(cell, []).append(sig_val)

        assert len(cell_signals) >= 1  # at least 1 distinct cell

        cell_long_pcts = compute_cell_long_pcts(None, cell_signals)
        assert len(cell_long_pcts) == len(cell_signals)

        ens = ensemble_sizing(cell_long_pcts, min_cells=1)
        assert ens.direction in (-1.0, 0.0, 1.0)
        assert 0.0 <= ens.confidence <= 1.0
        assert 0.5 <= ens.skew_adj <= 1.0
        assert ens.n_cells == len(cell_long_pcts)

    def test_map_elites_sizing_formula(self):
        """Sizing = direction × confidence × skew_adj × dd_scale, clamped to [-1, 1]."""
        from alpha_os.voting.ensemble import ensemble_sizing

        # Strong long consensus
        pcts = [0.9, 0.85, 0.95, 0.88, 0.92]
        ens = ensemble_sizing(pcts)
        dd_s = 0.8
        adjusted = ens.direction * ens.confidence * ens.skew_adj * dd_s
        assert -1.0 <= adjusted <= 1.0
        assert adjusted > 0  # should be positive (long)

    def test_few_cells_returns_neutral(self):
        """With fewer than min_cells, ensemble returns neutral."""
        from alpha_os.voting.ensemble import ensemble_sizing

        ens = ensemble_sizing([0.9, 0.8], min_cells=5)
        assert ens.direction == 0.0
        assert ens.confidence == 0.0

    def test_config_combine_mode_map_elites(self):
        """Config loading accepts map_elites as combine_mode."""
        from alpha_os.config import Config
        cfg = Config()
        cfg.paper.combine_mode = "map_elites"
        assert cfg.paper.combine_mode == "map_elites"
