"""Tests for forward testing — tracker and runner."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alpha_os.alpha.lifecycle import AlphaLifecycle, LifecycleConfig
from alpha_os.alpha.monitor import AlphaMonitor, MonitorConfig
from alpha_os.alpha.registry import AlphaRecord, AlphaRegistry, AlphaState
from alpha_os.config import Config
from alpha_os.data.store import DataStore
from alpha_os.forward.runner import ForwardConfig, ForwardCycleResult, ForwardRunner
from alpha_os.forward.tracker import ForwardTracker
from alpha_os.governance.audit_log import AuditLog


# ---------------------------------------------------------------------------
# ForwardTracker
# ---------------------------------------------------------------------------


class TestForwardTracker:
    def test_register_and_get_start_date(self, tmp_path):
        tracker = ForwardTracker(db_path=tmp_path / "fwd.db")
        tracker.register_alpha("a1", "2025-01-01")
        assert tracker.get_start_date("a1") == "2025-01-01"
        assert tracker.get_start_date("nonexistent") is None
        tracker.close()

    def test_record_and_get_returns(self, tmp_path):
        tracker = ForwardTracker(db_path=tmp_path / "fwd.db")
        tracker.register_alpha("a1", "2025-01-01")
        tracker.record("a1", "2025-01-02", 0.5, 0.01)
        tracker.record("a1", "2025-01-03", -0.3, -0.005)
        returns = tracker.get_returns("a1")
        assert len(returns) == 2
        assert returns[0] == pytest.approx(0.01)
        assert returns[1] == pytest.approx(-0.005)
        tracker.close()

    def test_cumulative_return(self, tmp_path):
        tracker = ForwardTracker(db_path=tmp_path / "fwd.db")
        tracker.register_alpha("a1", "2025-01-01")
        tracker.record("a1", "2025-01-02", 0.5, 0.01)
        tracker.record("a1", "2025-01-03", 0.5, 0.02)
        records = tracker.get_records("a1")
        assert len(records) == 2
        assert records[0].cumulative_return == pytest.approx(1.01)
        assert records[1].cumulative_return == pytest.approx(1.01 * 1.02)
        tracker.close()

    def test_get_last_date(self, tmp_path):
        tracker = ForwardTracker(db_path=tmp_path / "fwd.db")
        tracker.register_alpha("a1", "2025-01-01")
        assert tracker.get_last_date("a1") is None
        tracker.record("a1", "2025-01-02", 0.5, 0.01)
        tracker.record("a1", "2025-01-03", 0.5, 0.02)
        assert tracker.get_last_date("a1") == "2025-01-03"
        tracker.close()

    def test_summary_unregistered(self, tmp_path):
        tracker = ForwardTracker(db_path=tmp_path / "fwd.db")
        assert tracker.summary("nonexistent") is None
        tracker.close()

    def test_summary_empty(self, tmp_path):
        tracker = ForwardTracker(db_path=tmp_path / "fwd.db")
        tracker.register_alpha("a1", "2025-01-01")
        summary = tracker.summary("a1")
        assert summary is not None
        assert summary.n_days == 0
        assert summary.sharpe == 0.0
        tracker.close()

    def test_summary_with_data(self, tmp_path):
        tracker = ForwardTracker(db_path=tmp_path / "fwd.db")
        tracker.register_alpha("a1", "2025-01-01")
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.01, 50)
        for i, r in enumerate(rets):
            tracker.record("a1", f"2025-02-{i + 1:02d}", 0.5, float(r))
        summary = tracker.summary("a1")
        assert summary.n_days == 50
        assert summary.sharpe != 0.0
        assert summary.max_dd >= 0.0
        tracker.close()

    def test_tracked_alpha_ids_sorted(self, tmp_path):
        tracker = ForwardTracker(db_path=tmp_path / "fwd.db")
        tracker.register_alpha("b1", "2025-01-01")
        tracker.register_alpha("a1", "2025-01-01")
        ids = tracker.tracked_alpha_ids()
        assert ids == ["a1", "b1"]
        tracker.close()

    def test_idempotent_register(self, tmp_path):
        tracker = ForwardTracker(db_path=tmp_path / "fwd.db")
        tracker.register_alpha("a1", "2025-01-01")
        tracker.register_alpha("a1", "2025-02-01")
        assert tracker.get_start_date("a1") == "2025-01-01"
        tracker.close()

    def test_record_replaces_same_date(self, tmp_path):
        tracker = ForwardTracker(db_path=tmp_path / "fwd.db")
        tracker.register_alpha("a1", "2025-01-01")
        tracker.record("a1", "2025-01-02", 0.5, 0.01)
        tracker.record("a1", "2025-01-02", 0.6, 0.02)
        returns = tracker.get_returns("a1")
        assert len(returns) == 1
        assert returns[0] == pytest.approx(0.02)
        tracker.close()


# ---------------------------------------------------------------------------
# ForwardRunner integration (manual record → monitor → lifecycle)
# ---------------------------------------------------------------------------


class TestForwardRunnerIntegration:
    def test_cycle_result_dataclass(self):
        result = ForwardCycleResult(
            n_evaluated=5, n_degraded=1, n_rejected=0, n_restored=0,
            n_dormant=1, n_revived=0, elapsed=1.23,
        )
        assert result.n_evaluated == 5
        assert result.n_dormant == 1
        assert result.elapsed == pytest.approx(1.23)

    def test_healthy_alpha_stays_active(self, tmp_path):
        reg = AlphaRegistry(db_path=tmp_path / "reg.db")
        reg.register(AlphaRecord(
            alpha_id="healthy",
            expression="(neg f1)",
            state=AlphaState.ACTIVE,
            oos_sharpe=1.0,
        ))
        tracker = ForwardTracker(db_path=tmp_path / "fwd.db")
        monitor = AlphaMonitor(config=MonitorConfig(
            rolling_window=30, min_observations=5,
        ))
        lifecycle = AlphaLifecycle(reg, config=LifecycleConfig(
            probation_sharpe_min=0.3,
        ))

        tracker.register_alpha("healthy", "2025-01-01")
        rng = np.random.default_rng(42)
        rets = rng.normal(0.002, 0.005, 30)
        for i, r in enumerate(rets):
            tracker.record("healthy", f"2025-01-{i + 2:02d}", 0.5, float(r))

        all_returns = tracker.get_returns("healthy")
        monitor.record_batch("healthy", all_returns)
        status = monitor.check("healthy")
        assert status.rolling_sharpe > 0

        new_state = lifecycle.evaluate_active("healthy", status.rolling_sharpe)
        assert new_state == AlphaState.ACTIVE

        summary = tracker.summary("healthy")
        assert summary.n_days == 30
        assert summary.sharpe > 0

        tracker.close()
        reg.close()

    def test_degraded_alpha_to_probation(self, tmp_path):
        reg = AlphaRegistry(db_path=tmp_path / "reg.db")
        reg.register(AlphaRecord(
            alpha_id="bad",
            expression="(neg f1)",
            state=AlphaState.ACTIVE,
            oos_sharpe=1.0,
        ))
        tracker = ForwardTracker(db_path=tmp_path / "fwd.db")
        monitor = AlphaMonitor(config=MonitorConfig(
            rolling_window=30, sharpe_threshold=0.0, min_observations=10,
        ))
        lifecycle = AlphaLifecycle(reg, config=LifecycleConfig(
            probation_sharpe_min=0.0,
        ))

        tracker.register_alpha("bad", "2025-01-01")
        for i in range(30):
            tracker.record("bad", f"2025-01-{i + 2:02d}", -0.5, -0.005)

        all_returns = tracker.get_returns("bad")
        monitor.record_batch("bad", all_returns)
        status = monitor.check("bad")
        assert status.is_degraded
        assert status.rolling_sharpe < 0

        new_state = lifecycle.evaluate_active("bad", status.rolling_sharpe)
        assert new_state == AlphaState.PROBATION
        assert reg.get("bad").state == AlphaState.PROBATION

        tracker.close()
        reg.close()

    def test_probation_to_dormant(self, tmp_path):
        reg = AlphaRegistry(db_path=tmp_path / "reg.db")
        reg.register(AlphaRecord(
            alpha_id="dying",
            expression="(neg f1)",
            state=AlphaState.PROBATION,
        ))
        lifecycle = AlphaLifecycle(reg)

        new_state = lifecycle.evaluate_probation("dying", live_sharpe=-0.5)
        assert new_state == AlphaState.DORMANT

        reg.close()

    def test_probation_recovery_to_active(self, tmp_path):
        reg = AlphaRegistry(db_path=tmp_path / "reg.db")
        reg.register(AlphaRecord(
            alpha_id="recover",
            expression="(neg f1)",
            state=AlphaState.PROBATION,
        ))
        lifecycle = AlphaLifecycle(reg, config=LifecycleConfig(
            oos_sharpe_min=0.5,
        ))

        new_state = lifecycle.evaluate_probation("recover", live_sharpe=0.8)
        assert new_state == AlphaState.ACTIVE

        reg.close()

    def test_forward_config_defaults(self):
        cfg = ForwardConfig()
        assert cfg.check_interval == 14400
        assert cfg.min_forward_days == 30
        assert cfg.degradation_window == 63


# ---------------------------------------------------------------------------
# ForwardRunner.run_cycle integration test
# ---------------------------------------------------------------------------


class TestForwardRunnerCycle:
    @staticmethod
    def _populate_store(store, n_days=100, start="2025-01-01"):
        rng = np.random.default_rng(42)
        prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, n_days))
        dates = pd.bdate_range(start, periods=n_days)
        for i, d in enumerate(dates):
            store._conn.execute(
                "INSERT INTO signals (name, date, value) VALUES (?, ?, ?)",
                ("f1", d.strftime("%Y-%m-%d"), float(prices[i])),
            )
        store._conn.commit()
        return dates

    @staticmethod
    def _make_runner(tmp_path, store, reg):
        cfg = Config()
        runner = ForwardRunner(
            asset="f1", config=cfg,
            forward_config=ForwardConfig(),
            registry=reg,
            tracker=ForwardTracker(db_path=tmp_path / "fwd.db"),
            audit_log=AuditLog(tmp_path / "audit.jsonl"),
            store=store,
        )
        runner.features = ["f1"]
        runner.price_signal = "f1"
        return runner

    def test_run_cycle_evaluates_alphas(self, tmp_path):
        store = DataStore(tmp_path / "cache.db")
        dates = self._populate_store(store)
        # Use a date in the middle so matrix has enough rows
        fake_today = dates[50].date()

        reg = AlphaRegistry(db_path=tmp_path / "reg.db")
        reg.register(AlphaRecord(
            alpha_id="alpha_f1", expression="f1",
            state=AlphaState.ACTIVE, oos_sharpe=0.8,
        ))
        # Pre-register so start_date is early enough for data
        tracker = ForwardTracker(db_path=tmp_path / "fwd.db")
        tracker.register_alpha("alpha_f1", dates[0].strftime("%Y-%m-%d"))

        runner = self._make_runner(tmp_path, store, reg)
        runner.tracker = tracker
        with patch("alpha_os.forward.runner.date") as mock_date:
            mock_date.today.return_value = fake_today
            result = runner.run_cycle()
        assert isinstance(result, ForwardCycleResult)
        assert result.n_evaluated == 1
        assert result.elapsed > 0
        runner.close()

    def test_run_cycle_no_alphas(self, tmp_path):
        store = DataStore(tmp_path / "cache.db")
        dates = self._populate_store(store, n_days=50)
        last_date = dates[-1].date()

        reg = AlphaRegistry(db_path=tmp_path / "reg.db")
        runner = self._make_runner(tmp_path, store, reg)
        with patch("alpha_os.forward.runner.date") as mock_date:
            mock_date.today.return_value = last_date
            result = runner.run_cycle()
        assert result.n_evaluated == 0
        runner.close()

    def test_run_cycle_bad_alpha_counted(self, tmp_path):
        """An alpha with invalid expression should be counted as failed."""
        store = DataStore(tmp_path / "cache.db")
        dates = self._populate_store(store, n_days=50)
        last_date = dates[-1].date()

        reg = AlphaRegistry(db_path=tmp_path / "reg.db")
        reg.register(AlphaRecord(
            alpha_id="alpha_bad", expression="(neg missing_feature)",
            state=AlphaState.ACTIVE, oos_sharpe=0.5,
        ))

        runner = self._make_runner(tmp_path, store, reg)
        with patch("alpha_os.forward.runner.date") as mock_date:
            mock_date.today.return_value = last_date
            result = runner.run_cycle()
        assert result.n_evaluated == 0
        runner.close()
