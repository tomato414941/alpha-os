"""Tests for RegimeDetector."""
import numpy as np
import pytest

from alpha_os.alpha.monitor import RegimeDetector


@pytest.fixture
def detector():
    return RegimeDetector(short_window=21, long_window=63)


class TestRegimeDetector:
    def test_insufficient_data_returns_neutral(self, detector):
        r = np.random.default_rng(0).normal(0, 0.01, 30)
        status = detector.detect(r)
        assert status.current_vol_regime == "normal"
        assert status.drift_score == 0.0

    def test_high_vol_regime(self, detector):
        rng = np.random.default_rng(42)
        # Low vol for first 42 days, high vol for last 21
        low = rng.normal(0, 0.005, 42)
        high = rng.normal(0, 0.05, 21)
        r = np.concatenate([low, high])
        status = detector.detect(r)
        assert status.current_vol_regime == "high"
        assert status.vol_ratio > 1.5

    def test_low_vol_regime(self, detector):
        rng = np.random.default_rng(42)
        # High vol for first 42 days, low vol for last 21
        high = rng.normal(0, 0.05, 42)
        low = rng.normal(0, 0.005, 21)
        r = np.concatenate([high, low])
        status = detector.detect(r)
        assert status.current_vol_regime == "low"
        assert status.vol_ratio < 0.7

    def test_drift_detection(self, detector):
        rng = np.random.default_rng(42)
        # Two very different distributions
        first = rng.normal(0.05, 0.01, 32)
        second = rng.normal(-0.05, 0.01, 31)
        r = np.concatenate([first, second])
        status = detector.detect(r)
        assert status.drift_score > 0.3

    def test_no_drift_stable(self, detector):
        rng = np.random.default_rng(42)
        r = rng.normal(0, 0.01, 63)
        status = detector.detect(r)
        assert status.drift_score < 0.3

    def test_trending_regime(self, detector):
        # Positive autocorrelation: cumulative drift
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.01, 63)
        # Add momentum: each return partially follows previous
        trending = np.zeros(63)
        trending[0] = base[0]
        for i in range(1, 63):
            trending[i] = 0.5 * trending[i - 1] + base[i]
        status = detector.detect(trending)
        assert status.trend_regime == "trending"
