"""Tests for Path B: voting system."""
import numpy as np
import pytest

from alpha_os.voting.scorer import recency_weight, accuracy_weight
from alpha_os.voting.aggregator import vote_aggregate, VoteResult


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
