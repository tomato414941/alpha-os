"""Tests for Phase 4 testnet validation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from alpha_os.execution.executor import Fill
from alpha_os.validation.testnet import TestnetValidator


@dataclass
class _MockCycleResult:
    portfolio_value: float = 10000.0
    daily_pnl: float = 50.0
    daily_return: float = 0.005
    fills: list = None
    n_alphas_active: int = 5
    n_alphas_evaluated: int = 5

    def __post_init__(self):
        if self.fills is None:
            self.fills = []


def _mock_cb(halted=False, reason=""):
    cb = MagicMock()
    cb.halted = halted
    cb.halt_reason = reason
    return cb


class TestValidationState:
    def test_fresh_state(self, tmp_path):
        v = TestnetValidator(
            state_path=tmp_path / "state.json",
            report_path=tmp_path / "reports.jsonl",
        )
        assert v.state.consecutive_success_days == 0
        assert v.state.passed is False
        assert v.state.total_days_run == 0

    def test_consecutive_success_increments(self, tmp_path):
        v = TestnetValidator(
            state_path=tmp_path / "state.json",
            report_path=tmp_path / "reports.jsonl",
            target_days=3,
        )
        for day in range(1, 4):
            result = _MockCycleResult()
            recon = {"match": True, "qty_diff": 0.0, "cash_diff": 0.0}
            cb = _mock_cb()
            v.validate_cycle(result, recon, cb, [], today_override=f"2026-03-0{day}")

        assert v.state.consecutive_success_days == 3
        assert v.state.passed is True

    def test_error_resets_counter(self, tmp_path):
        v = TestnetValidator(
            state_path=tmp_path / "state.json",
            report_path=tmp_path / "reports.jsonl",
        )
        # Day 1: success
        v.validate_cycle(
            _MockCycleResult(), {"match": True}, _mock_cb(), [],
            today_override="2026-03-01",
        )
        assert v.state.consecutive_success_days == 1

        # Day 2: error (portfolio = 0)
        v.validate_cycle(
            _MockCycleResult(portfolio_value=0.0),
            {"match": True}, _mock_cb(), [],
            today_override="2026-03-02",
        )
        assert v.state.consecutive_success_days == 0

    def test_state_persists_across_instances(self, tmp_path):
        state_path = tmp_path / "state.json"
        report_path = tmp_path / "reports.jsonl"

        v1 = TestnetValidator(state_path=state_path, report_path=report_path)
        v1.validate_cycle(
            _MockCycleResult(), {"match": True}, _mock_cb(), [],
            today_override="2026-03-01",
        )
        assert v1.state.total_days_run == 1

        v2 = TestnetValidator(state_path=state_path, report_path=report_path)
        assert v2.state.total_days_run == 1
        assert v2.state.consecutive_success_days == 1

    def test_duplicate_day_not_double_counted(self, tmp_path):
        v = TestnetValidator(
            state_path=tmp_path / "state.json",
            report_path=tmp_path / "reports.jsonl",
        )
        for _ in range(3):
            v.validate_cycle(
                _MockCycleResult(), {"match": True}, _mock_cb(), [],
                today_override="2026-03-01",
            )
        assert v.state.total_days_run == 1
        assert v.state.consecutive_success_days == 1


class TestDailyReport:
    def test_reconciliation_mismatch_is_error(self, tmp_path):
        v = TestnetValidator(
            state_path=tmp_path / "state.json",
            report_path=tmp_path / "reports.jsonl",
        )
        recon = {"match": False, "qty_diff": 0.05, "cash_diff": 200.0, "status": "ok"}
        report = v.validate_cycle(
            _MockCycleResult(), recon, _mock_cb(), [],
            today_override="2026-03-01",
        )
        assert report.has_errors is True
        assert any("Reconciliation mismatch" in e for e in report.error_details)

    def test_extreme_slippage_is_error(self, tmp_path):
        v = TestnetValidator(
            state_path=tmp_path / "state.json",
            report_path=tmp_path / "reports.jsonl",
        )
        fills = [
            Fill(symbol="BTC", side="buy", qty=0.1, price=50000,
                 slippage_bps=80.0, latency_ms=300),
            Fill(symbol="BTC", side="buy", qty=0.1, price=50000,
                 slippage_bps=60.0, latency_ms=200),
        ]
        report = v.validate_cycle(
            _MockCycleResult(fills=fills), {"match": True}, _mock_cb(), fills,
            today_override="2026-03-01",
        )
        assert report.has_errors is True
        assert any("slippage" in e.lower() for e in report.error_details)
        assert report.mean_slippage_bps == pytest.approx(70.0)

    def test_normal_slippage_is_ok(self, tmp_path):
        v = TestnetValidator(
            state_path=tmp_path / "state.json",
            report_path=tmp_path / "reports.jsonl",
        )
        fills = [
            Fill(symbol="BTC", side="buy", qty=0.1, price=50000,
                 slippage_bps=5.0, latency_ms=100),
        ]
        report = v.validate_cycle(
            _MockCycleResult(fills=fills), {"match": True}, _mock_cb(), fills,
            today_override="2026-03-01",
        )
        assert report.has_errors is False

    def test_reports_appended_to_jsonl(self, tmp_path):
        report_path = tmp_path / "reports.jsonl"
        v = TestnetValidator(
            state_path=tmp_path / "state.json",
            report_path=report_path,
        )
        v.validate_cycle(
            _MockCycleResult(), {"match": True}, _mock_cb(), [],
            today_override="2026-03-01",
        )
        v.validate_cycle(
            _MockCycleResult(), {"match": True}, _mock_cb(), [],
            today_override="2026-03-02",
        )

        lines = report_path.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            r = json.loads(line)
            assert "date" in r
            assert "has_errors" in r

    def test_zero_portfolio_is_error(self, tmp_path):
        v = TestnetValidator(
            state_path=tmp_path / "state.json",
            report_path=tmp_path / "reports.jsonl",
        )
        report = v.validate_cycle(
            _MockCycleResult(portfolio_value=0.0),
            {"match": True}, _mock_cb(), [],
            today_override="2026-03-01",
        )
        assert report.has_errors is True
        assert any("zero" in e.lower() for e in report.error_details)


class TestOrderFailures:
    def test_order_failures_produce_error(self, tmp_path):
        """Order failures should be flagged as errors."""
        v = TestnetValidator(
            state_path=tmp_path / "state.json",
            report_path=tmp_path / "reports.jsonl",
        )
        report = v.validate_cycle(
            _MockCycleResult(), {"match": True}, _mock_cb(), [],
            order_failures=2,
            today_override="2026-03-01",
        )
        assert report.has_errors is True
        assert report.n_order_failures == 2
        assert any("Order failures" in e for e in report.error_details)

    def test_zero_order_failures_no_error(self, tmp_path):
        """Zero order failures should not produce an error."""
        v = TestnetValidator(
            state_path=tmp_path / "state.json",
            report_path=tmp_path / "reports.jsonl",
        )
        report = v.validate_cycle(
            _MockCycleResult(), {"match": True}, _mock_cb(), [],
            order_failures=0,
            today_override="2026-03-01",
        )
        assert report.n_order_failures == 0
        assert not any("Order failures" in e for e in report.error_details)

    def test_order_failures_reset_consecutive_days(self, tmp_path):
        """Order failures should reset the consecutive success counter."""
        v = TestnetValidator(
            state_path=tmp_path / "state.json",
            report_path=tmp_path / "reports.jsonl",
        )
        # Day 1: success
        v.validate_cycle(
            _MockCycleResult(), {"match": True}, _mock_cb(), [],
            today_override="2026-03-01",
        )
        assert v.state.consecutive_success_days == 1

        # Day 2: order failure
        v.validate_cycle(
            _MockCycleResult(), {"match": True}, _mock_cb(), [],
            order_failures=1,
            today_override="2026-03-02",
        )
        assert v.state.consecutive_success_days == 0


class TestFillFields:
    def test_fill_default_slippage_and_latency(self):
        fill = Fill(symbol="BTC", side="buy", qty=0.1, price=50000)
        assert fill.slippage_bps == 0.0
        assert fill.latency_ms == 0.0

    def test_fill_with_slippage_and_latency(self):
        fill = Fill(
            symbol="BTC", side="buy", qty=0.1, price=50000,
            slippage_bps=5.5, latency_ms=120.0,
        )
        assert fill.slippage_bps == 5.5
        assert fill.latency_ms == 120.0


class TestTrackerSlippageStats:
    def test_slippage_stats_with_fills(self, tmp_path):
        from alpha_os.paper.tracker import PaperPortfolioTracker

        tracker = PaperPortfolioTracker(db_path=tmp_path / "test.db")
        fills = [
            Fill(symbol="BTC", side="buy", qty=0.1, price=50000,
                 order_id="t-1", slippage_bps=5.0, latency_ms=100),
            Fill(symbol="BTC", side="buy", qty=0.1, price=50000,
                 order_id="t-2", slippage_bps=10.0, latency_ms=200),
        ]
        tracker.save_fills("2026-03-01", fills)

        stats = tracker.get_slippage_stats()
        assert stats["count"] == 2
        assert stats["mean_bps"] == pytest.approx(7.5)
        assert stats["max_bps"] == pytest.approx(10.0)

        lat_stats = tracker.get_latency_stats()
        assert lat_stats["count"] == 2
        assert lat_stats["mean_ms"] == pytest.approx(150.0)

        tracker.close()

    def test_empty_slippage_stats(self, tmp_path):
        from alpha_os.paper.tracker import PaperPortfolioTracker

        tracker = PaperPortfolioTracker(db_path=tmp_path / "test.db")
        stats = tracker.get_slippage_stats()
        assert stats["count"] == 0
        assert stats["mean_bps"] == 0.0
        tracker.close()
