"""Tests for Path B: voting system."""
import time

import numpy as np

from alpha_os.alpha.managed_alphas import AlphaRecord
from alpha_os.forward.tracker import ForwardRecord
from alpha_os.voting.scorer import recency_weight, accuracy_weight, accuracy_from_forward
from alpha_os.voting.aggregator import vote_aggregate
from alpha_os.voting.combiner import vote_combine


class TestRecencyWeight:
    def test_newer_gets_higher_weight(self):
        ages = np.array([0.0, 1.0, 7.0])
        w = recency_weight(ages, half_life=2.0)
        assert w[0] > w[1] > w[2]
        assert np.isclose(w.sum(), 1.0)

    def test_equal_ages(self):
        ages = np.array([3.0, 3.0, 3.0])
        w = recency_weight(ages, half_life=2.0)
        assert np.allclose(w, 1 / 3)

    def test_empty(self):
        w = recency_weight(np.array([]))
        assert len(w) == 0


class TestAccuracyWeight:
    def test_perfect_accuracy(self):
        signals = np.array([[1.0, 1.0, 1.0]])
        outcomes = np.array([1.0, 1.0, 1.0])
        acc = accuracy_weight(signals, outcomes, lookback=3)
        assert acc[0] == 1.0

    def test_zero_accuracy(self):
        signals = np.array([[-1.0, -1.0, -1.0]])
        outcomes = np.array([1.0, 1.0, 1.0])
        acc = accuracy_weight(signals, outcomes, lookback=3)
        assert acc[0] == 0.0

    def test_lookback_clips(self):
        signals = np.array([[1.0, -1.0, 1.0]])
        outcomes = np.array([1.0, 1.0, 1.0])
        acc = accuracy_weight(signals, outcomes, lookback=1)
        assert acc[0] == 1.0  # only last element


class TestAccuracyFromForward:
    def _make_record(self, signal: float, daily_return: float) -> ForwardRecord:
        return ForwardRecord(
            alpha_id="test", date="2026-01-01",
            signal_value=signal, daily_return=daily_return,
            cumulative_return=1.0,
        )

    def test_perfect(self):
        records = [
            self._make_record(1.0, 0.01),
            self._make_record(-1.0, -0.02),
            self._make_record(0.5, 0.005),
        ]
        assert accuracy_from_forward(records, lookback=3) == 1.0

    def test_wrong(self):
        records = [
            self._make_record(1.0, -0.01),
            self._make_record(-1.0, 0.02),
        ]
        assert accuracy_from_forward(records, lookback=5) == 0.0

    def test_empty(self):
        assert accuracy_from_forward([], lookback=5) == 0.5

    def test_lookback_limits(self):
        records = [
            self._make_record(1.0, -0.01),  # wrong (old, excluded)
            self._make_record(1.0, 0.01),    # correct
        ]
        assert accuracy_from_forward(records, lookback=1) == 1.0


class TestVoteAggregate:
    def test_unanimous_long(self):
        signals = {"a": 0.8, "b": 0.6, "c": 0.9, "d": 0.7, "e": 0.5}
        weights = {k: 0.2 for k in signals}
        result = vote_aggregate(signals, weights)
        assert result.direction == 1.0
        assert result.confidence > 0.9
        assert result.long_pct == 1.0
        assert result.short_pct == 0.0

    def test_split_vote(self):
        signals = {"a": 1.0, "b": -1.0, "c": 1.0, "d": -1.0, "e": 0.1}
        weights = {k: 0.2 for k in signals}
        result = vote_aggregate(signals, weights)
        assert result.confidence < 0.5  # close to split

    def test_below_min_voters(self):
        signals = {"a": 1.0, "b": 0.5}
        weights = {"a": 0.5, "b": 0.5}
        result = vote_aggregate(signals, weights, min_voters=5)
        assert result.direction == 0.0
        assert result.confidence == 0.0

    def test_empty(self):
        result = vote_aggregate({}, {})
        assert result.n_voters == 0


class TestVoteCombine:
    def test_with_mock_tracker(self, tmp_path):
        from alpha_os.forward.tracker import ForwardTracker

        tracker = ForwardTracker(db_path=tmp_path / "test_fwd.db")
        now = time.time()

        # Create alphas with different ages
        records = {}
        signals = {}
        for i, aid in enumerate(["a1", "a2", "a3", "a4", "a5"]):
            rec = AlphaRecord(
                alpha_id=aid, expression="x",
                created_at=now - (i + 1) * 86400,  # 1-5 days old
            )
            records[aid] = rec
            signals[aid] = 0.8 if i < 3 else -0.5  # 3 long, 2 short

            # Register and add forward records
            tracker.register_alpha(aid, "2026-01-01")
            for day in range(3):
                tracker.record(aid, f"2026-01-0{day+1}", 0.5, 0.01)

        result = vote_combine(signals, tracker, records, min_voters=3)
        assert result.n_voters == 5
        assert result.direction == 1.0  # majority long
        assert 0.0 < result.confidence <= 1.0
        tracker.close()

    def test_empty_signals(self, tmp_path):
        from alpha_os.forward.tracker import ForwardTracker
        tracker = ForwardTracker(db_path=tmp_path / "test_fwd2.db")
        result = vote_combine({}, tracker, {})
        assert result.n_voters == 0
        assert result.direction == 0.0
        tracker.close()
