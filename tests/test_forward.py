"""Tests for forward testing — tracker."""

import sqlite3

import numpy as np
import pytest

from alpha_os_recovery.forward.tracker import HypothesisObservationTracker


# ---------------------------------------------------------------------------
# HypothesisObservationTracker
# ---------------------------------------------------------------------------


class TestHypothesisObservationTracker:
    def test_register_and_get_start_date(self, tmp_path):
        tracker = HypothesisObservationTracker(db_path=tmp_path / "fwd.db")
        tracker.register_hypothesis("a1", "2025-01-01")
        assert tracker.get_hypothesis_start_date("a1") == "2025-01-01"
        assert tracker.get_hypothesis_start_date("nonexistent") is None
        tracker.close()

    def test_record_and_get_returns(self, tmp_path):
        tracker = HypothesisObservationTracker(db_path=tmp_path / "fwd.db")
        tracker.register_hypothesis("a1", "2025-01-01")
        tracker.record("a1", "2025-01-02", 0.5, 0.01)
        tracker.record("a1", "2025-01-03", -0.3, -0.005)
        returns = tracker.get_hypothesis_returns("a1")
        assert len(returns) == 2
        assert returns[0] == pytest.approx(0.01)
        assert returns[1] == pytest.approx(-0.005)
        tracker.close()

    def test_get_hypothesis_signal_history(self, tmp_path):
        tracker = HypothesisObservationTracker(db_path=tmp_path / "fwd.db")
        tracker.register_hypothesis("a1", "2025-01-01")
        tracker.record("a1", "2025-01-02", 0.5, 0.01)
        tracker.record("a1", "2025-01-03", -0.3, -0.005)
        tracker.record("a1", "2025-01-04", 0.1, 0.002)

        signals = tracker.get_hypothesis_signal_history("a1", limit=2)

        assert signals == pytest.approx([0.1, -0.3])
        tracker.close()

    def test_cumulative_return(self, tmp_path):
        tracker = HypothesisObservationTracker(db_path=tmp_path / "fwd.db")
        tracker.register_hypothesis("a1", "2025-01-01")
        tracker.record("a1", "2025-01-02", 0.5, 0.01)
        tracker.record("a1", "2025-01-03", 0.5, 0.02)
        records = tracker.get_hypothesis_records("a1")
        assert len(records) == 2
        assert records[0].cumulative_return == pytest.approx(1.01)
        assert records[1].cumulative_return == pytest.approx(1.01 * 1.02)
        tracker.close()

    def test_get_last_date(self, tmp_path):
        tracker = HypothesisObservationTracker(db_path=tmp_path / "fwd.db")
        tracker.register_hypothesis("a1", "2025-01-01")
        assert tracker.get_hypothesis_last_date("a1") is None
        tracker.record("a1", "2025-01-02", 0.5, 0.01)
        tracker.record("a1", "2025-01-03", 0.5, 0.02)
        assert tracker.get_hypothesis_last_date("a1") == "2025-01-03"
        tracker.close()

    def test_summary_unregistered(self, tmp_path):
        tracker = HypothesisObservationTracker(db_path=tmp_path / "fwd.db")
        assert tracker.hypothesis_summary("nonexistent") is None
        tracker.close()

    def test_summary_empty(self, tmp_path):
        tracker = HypothesisObservationTracker(db_path=tmp_path / "fwd.db")
        tracker.register_hypothesis("a1", "2025-01-01")
        summary = tracker.hypothesis_summary("a1")
        assert summary is not None
        assert summary.n_days == 0
        assert summary.sharpe == 0.0
        tracker.close()

    def test_summary_with_data(self, tmp_path):
        tracker = HypothesisObservationTracker(db_path=tmp_path / "fwd.db")
        tracker.register_hypothesis("a1", "2025-01-01")
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.01, 50)
        for i, r in enumerate(rets):
            tracker.record("a1", f"2025-02-{i + 1:02d}", 0.5, float(r))
        summary = tracker.hypothesis_summary("a1")
        assert summary.n_days == 50
        assert summary.sharpe != 0.0
        assert summary.max_dd >= 0.0
        tracker.close()

    def test_tracked_hypothesis_ids_sorted(self, tmp_path):
        tracker = HypothesisObservationTracker(db_path=tmp_path / "fwd.db")
        tracker.register_hypothesis("b1", "2025-01-01")
        tracker.register_hypothesis("a1", "2025-01-01")
        ids = tracker.tracked_hypothesis_ids()
        assert ids == ["a1", "b1"]
        tracker.close()

    def test_idempotent_register(self, tmp_path):
        tracker = HypothesisObservationTracker(db_path=tmp_path / "fwd.db")
        tracker.register_hypothesis("a1", "2025-01-01")
        tracker.register_hypothesis("a1", "2025-02-01")
        assert tracker.get_hypothesis_start_date("a1") == "2025-01-01"
        tracker.close()

    def test_record_replaces_same_date(self, tmp_path):
        tracker = HypothesisObservationTracker(db_path=tmp_path / "fwd.db")
        tracker.register_hypothesis("a1", "2025-01-01")
        tracker.record("a1", "2025-01-02", 0.5, 0.01)
        tracker.record("a1", "2025-01-02", 0.6, 0.02)
        returns = tracker.get_hypothesis_returns("a1")
        records = tracker.get_hypothesis_records("a1")
        assert len(returns) == 1
        assert returns[0] == pytest.approx(0.02)
        assert records[0].cumulative_return == pytest.approx(1.02)
        tracker.close()

    def test_record_replaces_same_date_after_prior_day_keeps_chain(self, tmp_path):
        tracker = HypothesisObservationTracker(db_path=tmp_path / "fwd.db")
        tracker.register_hypothesis("a1", "2025-01-01")
        tracker.record("a1", "2025-01-02", 0.5, 0.01)
        tracker.record("a1", "2025-01-03", 0.5, 0.02)
        tracker.record("a1", "2025-01-03", 0.6, -0.01)
        records = tracker.get_hypothesis_records("a1")
        assert len(records) == 2
        assert records[0].cumulative_return == pytest.approx(1.01)
        assert records[1].cumulative_return == pytest.approx(1.01 * 0.99)
        tracker.close()

    def test_legacy_forward_schema_is_migrated(self, tmp_path):
        db = tmp_path / "legacy_fwd.db"
        conn = sqlite3.connect(db)
        conn.execute(
            """
            CREATE TABLE forward_returns (
                alpha_id TEXT NOT NULL,
                date TEXT NOT NULL,
                signal_value REAL NOT NULL,
                daily_return REAL NOT NULL,
                cumulative_return REAL NOT NULL,
                recorded_at REAL NOT NULL,
                PRIMARY KEY (alpha_id, date)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE forward_meta (
                alpha_id TEXT PRIMARY KEY,
                forward_start_date TEXT NOT NULL,
                adopted_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO forward_meta (alpha_id, forward_start_date, adopted_at) VALUES (?, ?, ?)",
            ("h1", "2025-01-01", 1.0),
        )
        conn.execute(
            """
            INSERT INTO forward_returns
            (alpha_id, date, signal_value, daily_return, cumulative_return, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("h1", "2025-01-02", 0.5, 0.01, 1.01, 1.0),
        )
        conn.commit()
        conn.close()

        tracker = HypothesisObservationTracker(db_path=db)
        assert tracker.get_hypothesis_start_date("h1") == "2025-01-01"
        assert tracker.get_hypothesis_returns("h1") == pytest.approx([0.01])
        returns_columns = {
            row[1]
            for row in tracker._conn.execute("PRAGMA table_info(hypothesis_observations)")
        }
        meta_columns = {
            row[1]
            for row in tracker._conn.execute("PRAGMA table_info(hypothesis_observation_meta)")
        }
        assert "hypothesis_id" in returns_columns
        assert "hypothesis_id" in meta_columns
        tracker.close()

    def test_legacy_forward_file_name_is_renamed(self, tmp_path):
        legacy_db = tmp_path / "forward_returns.db"
        conn = sqlite3.connect(legacy_db)
        conn.execute(
            """
            CREATE TABLE forward_returns (
                hypothesis_id TEXT NOT NULL,
                date TEXT NOT NULL,
                signal_value REAL NOT NULL,
                daily_return REAL NOT NULL,
                cumulative_return REAL NOT NULL,
                recorded_at REAL NOT NULL,
                PRIMARY KEY (hypothesis_id, date)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE forward_meta (
                hypothesis_id TEXT PRIMARY KEY,
                forward_start_date TEXT NOT NULL,
                adopted_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO forward_meta (hypothesis_id, forward_start_date, adopted_at) VALUES (?, ?, ?)",
            ("h1", "2025-01-01", 1.0),
        )
        conn.execute(
            """
            INSERT INTO forward_returns
            (hypothesis_id, date, signal_value, daily_return, cumulative_return, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("h1", "2025-01-02", 0.5, 0.01, 1.01, 1.0),
        )
        conn.commit()
        conn.close()

        tracker = HypothesisObservationTracker(
            db_path=tmp_path / "hypothesis_observations.db"
        )
        assert legacy_db.exists() is False
        assert tracker.get_hypothesis_returns("h1") == pytest.approx([0.01])
        tracker.close()
