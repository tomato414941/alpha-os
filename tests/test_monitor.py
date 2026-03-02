"""Tests for alpha monitor, audit log, and scheduler."""
import json
import numpy as np

from alpha_os.alpha.monitor import AlphaMonitor, MonitorConfig, MonitorStatus
from alpha_os.governance.audit_log import AuditLog
from alpha_os.pipeline.scheduler import PipelineScheduler, SchedulerConfig


# ---------------------------------------------------------------------------
# Alpha Monitor
# ---------------------------------------------------------------------------

class TestAlphaMonitor:
    def test_record_and_check(self):
        mon = AlphaMonitor()
        rng = np.random.RandomState(42)
        rets = rng.normal(0.001, 0.01, 100)
        mon.record_batch("a1", rets)
        status = mon.check("a1")
        assert isinstance(status, MonitorStatus)
        assert status.alpha_id == "a1"
        assert status.rolling_sharpe != 0.0

    def test_no_data(self):
        mon = AlphaMonitor()
        status = mon.check("nonexistent")
        assert status.is_degraded is False
        assert status.rolling_sharpe == 0.0

    def test_insufficient_data(self):
        mon = AlphaMonitor(config=MonitorConfig(min_observations=50))
        for i in range(10):
            mon.record("a1", 0.001)
        status = mon.check("a1")
        assert status.is_degraded is False

    def test_degraded_negative_sharpe(self):
        mon = AlphaMonitor(config=MonitorConfig(
            rolling_window=30, sharpe_threshold=0.0, min_observations=20,
        ))
        # Consistently negative returns
        for _ in range(50):
            mon.record("a1", -0.005)
        status = mon.check("a1")
        assert status.is_degraded is True
        assert any("Sharpe" in r for r in status.degradation_reasons)

    def test_degraded_high_drawdown(self):
        mon = AlphaMonitor(config=MonitorConfig(
            rolling_window=30, drawdown_threshold=0.05, min_observations=20,
        ))
        # Large loss
        rets = [-0.03] * 30
        mon.record_batch("a1", rets)
        status = mon.check("a1")
        assert status.is_degraded is True
        assert any("MaxDD" in r for r in status.degradation_reasons)

    def test_healthy_alpha(self):
        mon = AlphaMonitor(config=MonitorConfig(
            rolling_window=63, sharpe_threshold=0.0, min_observations=20,
        ))
        rng = np.random.RandomState(42)
        rets = rng.normal(0.002, 0.005, 100)  # strong positive
        mon.record_batch("a1", rets)
        status = mon.check("a1")
        assert status.is_degraded is False
        assert status.rolling_sharpe > 0

    def test_check_all(self):
        mon = AlphaMonitor()
        rng = np.random.RandomState(42)
        mon.record_batch("a1", rng.normal(0.001, 0.01, 50))
        mon.record_batch("a2", rng.normal(-0.001, 0.01, 50))
        results = mon.check_all()
        assert len(results) == 2

    def test_clear(self):
        mon = AlphaMonitor()
        mon.record("a1", 0.01)
        mon.clear("a1")
        status = mon.check("a1")
        assert status.rolling_sharpe == 0.0


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_log_and_read(self, tmp_path):
        log = AuditLog(log_path=tmp_path / "audit.jsonl")
        log.log("test_event", alpha_id="a1", details={"key": "value"})
        events = log.read_all()
        assert len(events) == 1
        assert events[0].event_type == "test_event"
        assert events[0].alpha_id == "a1"
        assert events[0].details["key"] == "value"

    def test_append_only(self, tmp_path):
        log = AuditLog(log_path=tmp_path / "audit.jsonl")
        log.log("event1")
        log.log("event2")
        log.log("event3")
        events = log.read_all()
        assert len(events) == 3

    def test_log_state_change(self, tmp_path):
        log = AuditLog(log_path=tmp_path / "audit.jsonl")
        log.log_state_change("a1", "born", "active", reason="passed gates")
        events = log.read_all()
        assert events[0].details["old_state"] == "born"
        assert events[0].details["new_state"] == "active"

    def test_log_adoption(self, tmp_path):
        log = AuditLog(log_path=tmp_path / "audit.jsonl")
        log.log_adoption("a1", "(neg nvda)", {"sharpe": 1.5, "pbo": 0.2})
        events = log.read_all()
        assert events[0].event_type == "adoption"
        assert events[0].details["expression"] == "(neg nvda)"

    def test_log_trade(self, tmp_path):
        log = AuditLog(log_path=tmp_path / "audit.jsonl")
        log.log_trade("a1", "NVDA", "buy", 10.0, 150.0)
        events = log.read_all()
        assert events[0].details["symbol"] == "NVDA"
        assert events[0].details["qty"] == 10.0

    def test_read_by_type(self, tmp_path):
        log = AuditLog(log_path=tmp_path / "audit.jsonl")
        log.log("trade", alpha_id="a1")
        log.log("state_change", alpha_id="a1")
        log.log("trade", alpha_id="a2")
        trades = log.read_by_type("trade")
        assert len(trades) == 2

    def test_read_by_alpha(self, tmp_path):
        log = AuditLog(log_path=tmp_path / "audit.jsonl")
        log.log("event1", alpha_id="a1")
        log.log("event2", alpha_id="a2")
        log.log("event3", alpha_id="a1")
        a1_events = log.read_by_alpha("a1")
        assert len(a1_events) == 2

    def test_empty_log(self, tmp_path):
        log = AuditLog(log_path=tmp_path / "audit.jsonl")
        assert log.read_all() == []

    def test_jsonl_format(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(log_path=path)
        log.log("test")
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert "timestamp" in parsed
        assert "event_type" in parsed


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TestScheduler:
    def test_max_runs(self):
        call_count = 0

        def run():
            nonlocal call_count
            call_count += 1

        sched = PipelineScheduler(
            run_fn=run,
            config=SchedulerConfig(interval_seconds=0, max_runs=3),
        )
        sched.start()
        assert call_count == 3
        assert sched.run_count == 3

    def test_stop(self):
        call_count = 0

        def run():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                sched.stop()

        sched = PipelineScheduler(
            run_fn=run,
            config=SchedulerConfig(interval_seconds=0, max_runs=0),
        )
        sched.start()
        assert call_count == 2

    def test_error_retry(self):
        call_count = 0

        def run():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("test error")
            if call_count >= 3:
                sched.stop()

        sched = PipelineScheduler(
            run_fn=run,
            config=SchedulerConfig(interval_seconds=0, max_runs=0, retry_delay=0),
        )
        sched.start()
        assert call_count == 3  # 1 fail + 2 success
        assert sched.run_count == 2  # only successes counted

    def test_is_running(self):
        sched = PipelineScheduler(
            run_fn=lambda: None,
            config=SchedulerConfig(max_runs=1, interval_seconds=0),
        )
        assert sched.is_running is False
        sched.start()
        assert sched.is_running is False  # stopped after max_runs
