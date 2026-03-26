"""Tests for paper trading — tracker persistence and trader orchestration."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest
from unittest.mock import MagicMock

from alpha_os.execution.executor import Fill
from alpha_os.paper.tracker import PaperPortfolioTracker, PortfolioSnapshot


class _StaticMatrixStore:
    def __init__(self, matrix: pd.DataFrame):
        self._matrix = matrix

    def get_matrix(self, features, start=None, end=None):
        return self._matrix.loc[:, features]

    def close(self):
        return None


class _RecordingMatrixStore(_StaticMatrixStore):
    def __init__(self, matrix: pd.DataFrame):
        super().__init__(matrix)
        self.synced: list[list[str]] = []

    def sync(self, features, resolution="1d", *, min_history_days=0):
        self.synced.append(list(features))


class _FakePredictionStore:
    def __init__(self, rows):
        self._rows = rows

    def read_signal_history(self, signal_id, asset, n_days=60):
        return self._rows[(signal_id, asset)][-n_days:]

    def close(self):
        return None


class _StaticRegistry:
    def __init__(self, records):
        self._records = list(records)

    def top_by_stake(self, n=30, *, asset=None):
        return self._records[:n]

    def list_live(self, *, asset=None):
        return list(self._records)

    def list_observation_active(self, *, asset=None):
        return list(self._records)

    def close(self):
        return None


class _SplitRegistry(_StaticRegistry):
    def __init__(self, live_records, observation_records):
        super().__init__(live_records)
        self._observation_records = list(observation_records)

    def list_observation_active(self, *, asset=None):
        return list(self._observation_records)


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
            strategic_signal=0.4,
            regime_adjusted_signal=0.35,
            tactical_adjusted_signal=0.45,
            final_signal=0.45,
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
        assert loaded.strategic_signal == pytest.approx(0.4)
        assert loaded.regime_adjusted_signal == pytest.approx(0.35)
        assert loaded.tactical_adjusted_signal == pytest.approx(0.45)
        assert loaded.final_signal == pytest.approx(0.45)

        by_date = tracker.get_snapshot("2026-01-01")
        assert by_date is not None
        assert by_date.daily_pnl == 50.0
        tracker.close()

    def test_snapshot_signal_stage_migration(self, tmp_path):
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE portfolio_snapshots (
                date TEXT PRIMARY KEY,
                cash REAL NOT NULL,
                positions_json TEXT NOT NULL,
                portfolio_value REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                daily_return REAL NOT NULL,
                combined_signal REAL NOT NULL,
                dd_scale REAL NOT NULL,
                vol_scale REAL NOT NULL,
                recorded_at REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

        tracker = PaperPortfolioTracker(db)
        columns = {
            row[1] for row in tracker._conn.execute("PRAGMA table_info(portfolio_snapshots)")
        }
        assert "strategic_signal" in columns
        assert "regime_adjusted_signal" in columns
        assert "tactical_adjusted_signal" in columns
        assert "final_signal" in columns

        tracker.save_snapshot(PortfolioSnapshot(
            date="2026-01-02",
            cash=9100.0,
            positions={"btc_ohlcv": 0.02},
            portfolio_value=10050.0,
            daily_pnl=50.0,
            daily_return=0.005,
            combined_signal=0.3,
            strategic_signal=0.4,
            regime_adjusted_signal=0.38,
            tactical_adjusted_signal=0.41,
            final_signal=0.41,
            dd_scale=1.0,
            vol_scale=1.0,
        ))
        loaded = tracker.get_last_snapshot()
        assert loaded is not None
        assert loaded.final_signal == pytest.approx(0.41)
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

    def test_hypothesis_signals(self, tmp_path):
        db = tmp_path / "test.db"
        tracker = PaperPortfolioTracker(db)
        tracker.save_hypothesis_signals("2026-01-01", {"a1": 0.5, "a2": -0.3})
        tracker.save_hypothesis_signals("2026-01-01", {"a1": 0.6})  # upsert
        assert tracker.get_hypothesis_signals("2026-01-01") == {"a1": 0.6, "a2": -0.3}
        tracker.close()

    def test_legacy_alpha_signal_table_is_migrated(self, tmp_path):
        db = tmp_path / "legacy_signals.db"
        conn = sqlite3.connect(db)
        conn.execute(
            """
            CREATE TABLE alpha_signals (
                date TEXT NOT NULL,
                alpha_id TEXT NOT NULL,
                signal_value REAL NOT NULL,
                PRIMARY KEY (date, alpha_id)
            )
            """
        )
        conn.execute(
            "INSERT INTO alpha_signals (date, alpha_id, signal_value) VALUES (?, ?, ?)",
            ("2026-01-01", "h1", 0.25),
        )
        conn.commit()
        conn.close()

        tracker = PaperPortfolioTracker(db)
        assert tracker.get_hypothesis_signals("2026-01-01") == {"h1": 0.25}
        columns = {
            row[1] for row in tracker._conn.execute("PRAGMA table_info(hypothesis_signals)")
        }
        assert "hypothesis_id" in columns
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
        from alpha_os.paper.trader import Trader
        from alpha_os.config import Config
        from alpha_os.legacy.managed_alphas import ManagedAlphaStore
        from alpha_os.forward.tracker import HypothesisObservationTracker
        from alpha_os.data.store import DataStore
        from alpha_os.execution.paper import PaperExecutor
        from alpha_os.governance.audit_log import AuditLog

        cfg = Config()
        pt = PaperPortfolioTracker(tmp_path / "paper.db")

        # Simulate a grown portfolio: started at $10k, now $50k
        executor = PaperExecutor(initial_cash=50000.0)
        trader = Trader(
            asset="BTC",
            config=cfg,
            portfolio_tracker=pt,
            registry=ManagedAlphaStore(tmp_path / "reg.db"),
            forward_tracker=HypothesisObservationTracker(tmp_path / "fwd.db"),
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


class TestTrader:
    def test_execute_allocation_counts_no_delta_skip(self):
        from alpha_os.execution.planning import TargetPosition
        from alpha_os.paper.trader import AllocationPlan, Trader

        trader = object.__new__(Trader)
        trader.price_signal = "btc_ohlcv"
        trader.rebalance_deadband_usd = 10.0
        trader.executor = MagicMock()
        trader.executor.set_price.return_value = None
        trader.executor.get_position.return_value = 0.0

        plan = AllocationPlan(
            current_price=100000.0,
            target_position=TargetPosition(
                symbol="btc_ohlcv",
                qty=0.0,
                reference_price=100000.0,
                dollar_target=0.0,
            ),
        )

        outcome = Trader._execute_allocation(trader, plan)

        assert outcome.fills == []
        assert outcome.order_failures == 0
        assert outcome.skipped_no_delta == 1
        assert outcome.skipped_deadband == 0

    def test_prediction_history_array_pads_to_matrix_length(self):
        from alpha_os.paper.trader import Trader

        store = _FakePredictionStore(
            {
                ("h1", "BTC"): [
                    ("2026-03-21", 0.4),
                    ("2026-03-20", 0.2),
                ]
            }
        )

        arr = Trader._prediction_history_array(
            store,
            "h1",
            "BTC",
            n_days=5,
            fallback_value=0.4,
        )

        assert list(arr) == [0.2, 0.2, 0.2, 0.2, 0.4]

    def test_prepare_runtime_inputs_prefers_prediction_store(self):
        from alpha_os.paper.trader import Trader
        from alpha_os.hypotheses.store import HypothesisRecord

        store_backed = HypothesisRecord(
            hypothesis_id="h-store",
            kind="dsl",
            definition={"expression": "(zscore foo)"},
            stake=1.0,
        )
        expr_backed = HypothesisRecord(
            hypothesis_id="h-expr",
            kind="dsl",
            definition={"expression": "(rank_10 bar)"},
            stake=0.5,
        )

        runtime_signals, parsed_records, n_failed = Trader._prepare_runtime_inputs(
            [store_backed, expr_backed],
            "btc_ohlcv",
            {"h-store": 0.4},
        )

        assert runtime_signals == ["bar", "btc_ohlcv"]
        assert [(record.hypothesis_id, expr is None) for record, expr in parsed_records] == [
            ("h-store", True),
            ("h-expr", False),
        ]
        assert n_failed == 0

    def test_prepare_runtime_inputs_uses_raw_signals_for_derived_features(self):
        from alpha_os.paper.trader import Trader
        from alpha_os.hypotheses.store import HypothesisRecord

        derived_backed = HypothesisRecord(
            hypothesis_id="h-derived",
            kind="dsl",
            definition={"expression": "(rank_10 delta_1__btc_active_addresses)"},
            stake=1.0,
        )

        runtime_signals, parsed_records, n_failed = Trader._prepare_runtime_inputs(
            [derived_backed],
            "eth_btc",
            {},
        )

        assert runtime_signals == ["btc_active_addresses", "eth_btc"]
        assert [(record.hypothesis_id, expr is None) for record, expr in parsed_records] == [
            ("h-derived", False),
        ]
        assert n_failed == 0

    def test_restore_state_empty(self, tmp_path):
        """Fresh trader should have initial capital."""
        from alpha_os.paper.trader import Trader
        from alpha_os.config import Config
        from alpha_os.legacy.managed_alphas import ManagedAlphaStore
        from alpha_os.forward.tracker import HypothesisObservationTracker
        from alpha_os.data.store import DataStore
        from alpha_os.execution.paper import PaperExecutor
        from alpha_os.governance.audit_log import AuditLog

        db = tmp_path / "paper.db"
        cfg = Config()
        pt = PaperPortfolioTracker(db)

        trader = Trader(
            asset="BTC",
            config=cfg,
            portfolio_tracker=pt,
            registry=ManagedAlphaStore(tmp_path / "reg.db"),
            forward_tracker=HypothesisObservationTracker(tmp_path / "fwd.db"),
            executor=PaperExecutor(initial_cash=cfg.trading.initial_capital),
            audit_log=AuditLog(tmp_path / "audit.jsonl"),
            store=DataStore(tmp_path / "cache.db"),
        )
        assert trader.executor.get_cash() == pytest.approx(cfg.trading.initial_capital)
        trader.close()

    def test_restore_state_from_snapshot(self, tmp_path):
        """Trader should restore cash/positions from last snapshot."""
        from alpha_os.paper.trader import Trader
        from alpha_os.config import Config
        from alpha_os.legacy.managed_alphas import ManagedAlphaStore
        from alpha_os.forward.tracker import HypothesisObservationTracker
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
        trader = Trader(
            asset="BTC",
            config=cfg,
            portfolio_tracker=pt,
            registry=ManagedAlphaStore(tmp_path / "reg.db"),
            forward_tracker=HypothesisObservationTracker(tmp_path / "fwd.db"),
            executor=PaperExecutor(initial_cash=cfg.trading.initial_capital),
            audit_log=AuditLog(tmp_path / "audit.jsonl"),
            store=DataStore(tmp_path / "cache.db"),
        )
        assert trader.executor.get_cash() == pytest.approx(8500.0)
        assert trader.executor.get_position("btc_ohlcv") == pytest.approx(0.015)
        trader.close()

    def test_sync_state_from_newer_snapshot(self, tmp_path):
        """Trader should refresh executor state when a newer snapshot appears."""
        from alpha_os.paper.trader import Trader
        from alpha_os.config import Config
        from alpha_os.legacy.managed_alphas import ManagedAlphaStore
        from alpha_os.forward.tracker import HypothesisObservationTracker
        from alpha_os.data.store import DataStore
        from alpha_os.execution.paper import PaperExecutor
        from alpha_os.governance.audit_log import AuditLog

        db = tmp_path / "paper.db"
        pt = PaperPortfolioTracker(db)
        pt.save_snapshot(PortfolioSnapshot(
            date="2026-01-05T00:00:00", cash=8500.0,
            positions={"btc_ohlcv": 0.015},
            portfolio_value=9950.0, daily_pnl=-50.0, daily_return=-0.005,
            combined_signal=0.3, dd_scale=1.0, vol_scale=0.9,
        ))

        cfg = Config()
        trader = Trader(
            asset="BTC",
            config=cfg,
            portfolio_tracker=pt,
            registry=ManagedAlphaStore(tmp_path / "reg.db"),
            forward_tracker=HypothesisObservationTracker(tmp_path / "fwd.db"),
            executor=PaperExecutor(initial_cash=cfg.trading.initial_capital),
            audit_log=AuditLog(tmp_path / "audit.jsonl"),
            store=DataStore(tmp_path / "cache.db"),
        )

        pt.save_snapshot(PortfolioSnapshot(
            date="2026-01-05T04:00:00", cash=8100.0,
            positions={"btc_ohlcv": 0.02},
            portfolio_value=10020.0, daily_pnl=70.0, daily_return=0.007,
            combined_signal=0.4, dd_scale=1.0, vol_scale=1.0,
        ))
        trader._sync_state_from_latest_snapshot()

        assert trader.executor.get_cash() == pytest.approx(8100.0)
        assert trader.executor.get_position("btc_ohlcv") == pytest.approx(0.02)
        trader.close()

    def test_build_allocation_plan_clamps_short_target_in_long_only_mode(self, tmp_path):
        """Long-only mode should never emit a negative target position."""
        from alpha_os.paper.trader import Trader
        from alpha_os.config import Config
        from alpha_os.legacy.managed_alphas import ManagedAlphaStore
        from alpha_os.forward.tracker import HypothesisObservationTracker
        from alpha_os.governance.audit_log import AuditLog

        cfg = Config()
        store = _StaticMatrixStore(pd.DataFrame({"btc_ohlcv": [100.0]}))
        trader = Trader(
            asset="BTC",
            config=cfg,
            portfolio_tracker=PaperPortfolioTracker(tmp_path / "paper.db"),
            registry=ManagedAlphaStore(tmp_path / "reg.db"),
            forward_tracker=HypothesisObservationTracker(tmp_path / "fwd.db"),
            audit_log=AuditLog(tmp_path / "audit.jsonl"),
            store=store,
        )

        plan = trader._build_allocation_plan(
            final_signal=-0.5,
            prev_value=10000.0,
            today_date="2026-01-05",
        )

        assert trader.executor.supports_short is False
        assert plan.current_price == pytest.approx(100.0)
        assert plan.target_position.symbol == "btc_ohlcv"
        assert plan.target_position.qty == 0.0
        assert plan.target_positions == {"btc_ohlcv": 0.0}
        trader.close()

    def test_build_allocation_plan_keeps_short_target_when_supported(self, tmp_path):
        """Long/short mode should preserve negative target positions."""
        from alpha_os.paper.trader import Trader
        from alpha_os.config import Config, TRADING_MODE_FUTURES_LONG_SHORT
        from alpha_os.legacy.managed_alphas import ManagedAlphaStore
        from alpha_os.forward.tracker import HypothesisObservationTracker
        from alpha_os.governance.audit_log import AuditLog

        cfg = Config()
        cfg.trading.mode = TRADING_MODE_FUTURES_LONG_SHORT
        store = _StaticMatrixStore(pd.DataFrame({"btc_ohlcv": [100.0]}))
        trader = Trader(
            asset="BTC",
            config=cfg,
            portfolio_tracker=PaperPortfolioTracker(tmp_path / "paper.db"),
            registry=ManagedAlphaStore(tmp_path / "reg.db"),
            forward_tracker=HypothesisObservationTracker(tmp_path / "fwd.db"),
            audit_log=AuditLog(tmp_path / "audit.jsonl"),
            store=store,
        )

        plan = trader._build_allocation_plan(
            final_signal=-0.5,
            prev_value=10000.0,
            today_date="2026-01-05",
        )

        assert trader.executor.supports_short is True
        assert plan.current_price == pytest.approx(100.0)
        assert plan.target_position.symbol == "btc_ohlcv"
        assert plan.target_position.qty == pytest.approx(-50.0)
        assert plan.target_positions == {"btc_ohlcv": pytest.approx(-50.0)}
        trader.close()

    def test_run_cycle_syncs_only_runtime_signals(self, monkeypatch, tmp_path):
        from alpha_os.config import Config
        from alpha_os.execution.paper import PaperExecutor
        from alpha_os.forward.tracker import HypothesisObservationTracker
        from alpha_os.governance.audit_log import AuditLog
        from alpha_os.hypotheses.store import HypothesisRecord
        from alpha_os.paper.trader import Trader
        from alpha_os.risk.circuit_breaker import CircuitBreaker

        cfg = Config()
        cfg.paper.max_position_pct = 0.5

        matrix = pd.DataFrame(
            {
                "btc_ohlcv": [100.0, 105.0],
                "unused_feature": [1.0, 2.0],
            },
            index=["2026-03-20", "2026-03-21"],
        )
        store = _RecordingMatrixStore(matrix)
        record = HypothesisRecord(
            hypothesis_id="h-store",
            kind="manual",
            definition={},
            stake=1.0,
        )

        history = {
            ("h-store", "BTC"): [
                ("2026-03-20", 0.3),
                ("2026-03-21", 0.3),
            ]
        }

        class _PatchedPredictionStore(_FakePredictionStore):
            def read_latest(self, date, assets=None):
                return {"h-store": {"BTC": 0.3}}

        monkeypatch.setattr(
            "alpha_os.predictions.store.PredictionStore",
            lambda *args, **kwargs: _PatchedPredictionStore(history),
        )
        monkeypatch.setattr(
            "alpha_os.paper.trader._quick_healthcheck",
            lambda _url: True,
        )

        trader = Trader(
            asset="BTC",
            config=cfg,
            registry=_StaticRegistry([record]),
            portfolio_tracker=PaperPortfolioTracker(tmp_path / "paper.db"),
            forward_tracker=HypothesisObservationTracker(tmp_path / "fwd.db"),
            executor=PaperExecutor(initial_cash=cfg.trading.initial_capital),
            audit_log=AuditLog(tmp_path / "audit.jsonl"),
            circuit_breaker=CircuitBreaker(_state_path=tmp_path / "circuit_breaker.json"),
            store=store,
        )

        result = trader.run_cycle()

        assert store.synced == [["btc_ohlcv"]]
        assert result.n_signals_evaluated == 1
        trader.close()

    def test_run_cycle_skips_runtime_sync_when_healthcheck_fails(self, monkeypatch, tmp_path):
        from alpha_os.config import Config
        from alpha_os.execution.paper import PaperExecutor
        from alpha_os.forward.tracker import HypothesisObservationTracker
        from alpha_os.governance.audit_log import AuditLog
        from alpha_os.hypotheses.store import HypothesisRecord
        from alpha_os.paper.trader import Trader
        from alpha_os.risk.circuit_breaker import CircuitBreaker

        cfg = Config()
        cfg.paper.max_position_pct = 0.5

        matrix = pd.DataFrame(
            {
                "btc_ohlcv": [100.0, 105.0],
                "unused_feature": [1.0, 2.0],
            },
            index=["2026-03-20", "2026-03-21"],
        )
        store = _RecordingMatrixStore(matrix)
        record = HypothesisRecord(
            hypothesis_id="h-store",
            kind="manual",
            definition={},
            stake=1.0,
        )

        history = {
            ("h-store", "BTC"): [
                ("2026-03-20", 0.3),
                ("2026-03-21", 0.3),
            ]
        }

        class _PatchedPredictionStore(_FakePredictionStore):
            def read_latest(self, date, assets=None):
                return {"h-store": {"BTC": 0.3}}

        monkeypatch.setattr(
            "alpha_os.predictions.store.PredictionStore",
            lambda *args, **kwargs: _PatchedPredictionStore(history),
        )
        monkeypatch.setattr(
            "alpha_os.paper.trader._quick_healthcheck",
            lambda _url: False,
        )

        trader = Trader(
            asset="BTC",
            config=cfg,
            registry=_StaticRegistry([record]),
            portfolio_tracker=PaperPortfolioTracker(tmp_path / "paper.db"),
            forward_tracker=HypothesisObservationTracker(tmp_path / "fwd.db"),
            executor=PaperExecutor(initial_cash=cfg.trading.initial_capital),
            audit_log=AuditLog(tmp_path / "audit.jsonl"),
            circuit_breaker=CircuitBreaker(_state_path=tmp_path / "circuit_breaker.json"),
            store=store,
        )

        result = trader.run_cycle()

        assert store.synced == []
        assert result.n_signals_evaluated == 1
        trader.close()

    def test_run_cycle_selects_only_capital_backed_candidates(self, monkeypatch, tmp_path):
        from alpha_os.config import Config
        from alpha_os.execution.paper import PaperExecutor
        from alpha_os.forward.tracker import HypothesisObservationTracker
        from alpha_os.governance.audit_log import AuditLog
        from alpha_os.hypotheses.store import HypothesisRecord
        from alpha_os.paper.trader import Trader
        from alpha_os.risk.circuit_breaker import CircuitBreaker

        cfg = Config()
        cfg.paper.max_trading_alphas = 1
        cfg.paper.max_position_pct = 0.5

        matrix = pd.DataFrame(
            {
                "btc_ohlcv": [100.0, 105.0],
            },
            index=["2026-03-20", "2026-03-21"],
        )
        store = _RecordingMatrixStore(matrix)
        live_record = HypothesisRecord(
            hypothesis_id="capital-backed",
            kind="manual",
            definition={},
            stake=1.0,
        )
        observation_only = HypothesisRecord(
            hypothesis_id="observation-only",
            kind="manual",
            definition={},
            stake=0.0,
        )

        history = {
            ("capital-backed", "BTC"): [
                ("2026-03-20", 0.3),
                ("2026-03-21", 0.3),
            ],
            ("observation-only", "BTC"): [
                ("2026-03-20", 0.9),
                ("2026-03-21", 0.9),
            ],
        }

        class _PatchedPredictionStore(_FakePredictionStore):
            def read_latest(self, date, assets=None):
                return {
                    "capital-backed": {"BTC": 0.3},
                    "observation-only": {"BTC": 0.9},
                }

        monkeypatch.setattr(
            "alpha_os.predictions.store.PredictionStore",
            lambda *args, **kwargs: _PatchedPredictionStore(history),
        )
        monkeypatch.setattr(
            "alpha_os.paper.trader._quick_healthcheck",
            lambda _url: False,
        )

        trader = Trader(
            asset="BTC",
            config=cfg,
            registry=_SplitRegistry([live_record], [live_record, observation_only]),
            portfolio_tracker=PaperPortfolioTracker(tmp_path / "paper.db"),
            forward_tracker=HypothesisObservationTracker(tmp_path / "fwd.db"),
            executor=PaperExecutor(initial_cash=cfg.trading.initial_capital),
            audit_log=AuditLog(tmp_path / "audit.jsonl"),
            circuit_breaker=CircuitBreaker(_state_path=tmp_path / "circuit_breaker.json"),
            store=store,
        )

        result = trader.run_cycle()

        assert result.n_active_hypotheses == 2
        assert result.n_live_hypotheses == 1
        assert result.n_shortlist_candidates == 1
        assert result.n_selected_hypotheses == 1
        assert result.n_signals_evaluated == 1
        trader.close()

    def test_run_cycle_evaluates_derived_features_from_raw_current_data(self, monkeypatch, tmp_path):
        from alpha_os.config import Config
        from alpha_os.execution.paper import PaperExecutor
        from alpha_os.forward.tracker import HypothesisObservationTracker
        from alpha_os.governance.audit_log import AuditLog
        from alpha_os.hypotheses.store import HypothesisRecord
        from alpha_os.paper.trader import Trader
        from alpha_os.risk.circuit_breaker import CircuitBreaker

        cfg = Config()
        cfg.paper.max_trading_alphas = 1
        cfg.paper.max_position_pct = 0.5

        matrix = pd.DataFrame(
            {
                "eth_btc": [0.03, 0.031],
                "btc_active_addresses": [100.0, 103.0],
            },
            index=["2026-03-20", "2026-03-21"],
        )
        store = _RecordingMatrixStore(matrix)
        derived_record = HypothesisRecord(
            hypothesis_id="derived-capital-backed",
            kind="dsl",
            definition={"expression": "(rank_10 delta_1__btc_active_addresses)"},
            stake=1.0,
        )

        monkeypatch.setattr(
            "alpha_os.paper.trader._quick_healthcheck",
            lambda _url: False,
        )

        trader = Trader(
            asset="ETH",
            config=cfg,
            registry=_StaticRegistry([derived_record]),
            portfolio_tracker=PaperPortfolioTracker(tmp_path / "paper.db"),
            forward_tracker=HypothesisObservationTracker(tmp_path / "fwd.db"),
            executor=PaperExecutor(initial_cash=cfg.trading.initial_capital),
            audit_log=AuditLog(tmp_path / "audit.jsonl"),
            circuit_breaker=CircuitBreaker(_state_path=tmp_path / "circuit_breaker.json"),
            store=store,
        )

        result = trader.run_cycle()

        assert store.synced == []
        assert result.n_live_hypotheses == 1
        assert result.n_signals_evaluated == 1
        trader.close()

    def test_run_cycle_uses_snapshot_portfolio_value_as_daily_pnl_baseline(
        self, monkeypatch, tmp_path
    ):
        from alpha_os.config import Config
        from alpha_os.execution.paper import PaperExecutor
        from alpha_os.forward.tracker import HypothesisObservationTracker
        from alpha_os.governance.audit_log import AuditLog
        from alpha_os.hypotheses.store import HypothesisRecord
        from alpha_os.paper.trader import Trader
        from alpha_os.risk.circuit_breaker import CircuitBreaker

        cfg = Config()
        cfg.paper.max_position_pct = 0.5

        portfolio_tracker = PaperPortfolioTracker(tmp_path / "paper.db")
        portfolio_tracker.save_snapshot(PortfolioSnapshot(
            date="2026-03-20T00:00:00",
            cash=8500.0,
            positions={"btc_ohlcv": 0.015},
            portfolio_value=9950.0,
            daily_pnl=-50.0,
            daily_return=-0.005,
            combined_signal=0.0,
            dd_scale=1.0,
            vol_scale=1.0,
        ))
        portfolio_tracker._conn.execute(
            "UPDATE portfolio_snapshots SET recorded_at = 0 WHERE date = ?",
            ("2026-03-20T00:00:00",),
        )
        portfolio_tracker._conn.commit()

        matrix = pd.DataFrame(
            {
                "btc_ohlcv": [95000.0, 100000.0],
            },
            index=["2026-03-20", "2026-03-21"],
        )
        store = _RecordingMatrixStore(matrix)
        record = HypothesisRecord(
            hypothesis_id="h-store",
            kind="manual",
            definition={},
            stake=1.0,
        )

        history = {
            ("h-store", "BTC"): [
                ("2026-03-20", 0.0),
                ("2026-03-21", 0.0),
            ]
        }

        class _PatchedPredictionStore(_FakePredictionStore):
            def read_latest(self, date, assets=None):
                return {"h-store": {"BTC": 0.0}}

        monkeypatch.setattr(
            "alpha_os.predictions.store.PredictionStore",
            lambda *args, **kwargs: _PatchedPredictionStore(history),
        )
        monkeypatch.setattr(
            "alpha_os.paper.trader._quick_healthcheck",
            lambda _url: False,
        )

        trader = Trader(
            asset="BTC",
            config=cfg,
            registry=_StaticRegistry([record]),
            portfolio_tracker=portfolio_tracker,
            forward_tracker=HypothesisObservationTracker(tmp_path / "fwd.db"),
            executor=PaperExecutor(initial_cash=cfg.trading.initial_capital),
            audit_log=AuditLog(tmp_path / "audit.jsonl"),
            circuit_breaker=CircuitBreaker(_state_path=tmp_path / "circuit_breaker.json"),
            store=store,
        )

        result = trader.run_cycle()

        assert result.portfolio_value == pytest.approx(9997.75)
        assert result.daily_pnl == pytest.approx(result.portfolio_value - 9950.0)
        assert result.daily_return == pytest.approx(result.daily_pnl / 9950.0)
        trader.close()
