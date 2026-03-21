"""Tests for forward testing — tracker."""

import numpy as np
import pytest

from alpha_os.forward.tracker import ForwardTracker


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
        records = tracker.get_records("a1")
        assert len(returns) == 1
        assert returns[0] == pytest.approx(0.02)
        assert records[0].cumulative_return == pytest.approx(1.02)
        tracker.close()

    def test_record_replaces_same_date_after_prior_day_keeps_chain(self, tmp_path):
        tracker = ForwardTracker(db_path=tmp_path / "fwd.db")
        tracker.register_alpha("a1", "2025-01-01")
        tracker.record("a1", "2025-01-02", 0.5, 0.01)
        tracker.record("a1", "2025-01-03", 0.5, 0.02)
        tracker.record("a1", "2025-01-03", 0.6, -0.01)
        records = tracker.get_records("a1")
        assert len(records) == 2
        assert records[0].cumulative_return == pytest.approx(1.01)
        assert records[1].cumulative_return == pytest.approx(1.01 * 0.99)
        tracker.close()
