from __future__ import annotations

import numpy as np
import pytest

from alpha_os.alpha.quality import (
    blend_quality,
    rolling_fitness,
)


def test_blend_quality_falls_back_to_prior_without_forward_observations():
    estimate = blend_quality(
        0.8,
        [],
        metric="sharpe",
        min_observations=20,
        full_weight_observations=63,
    )

    assert estimate.prior_quality == pytest.approx(0.8)
    assert estimate.live_quality == pytest.approx(0.0)
    assert estimate.blended_quality == pytest.approx(0.8)
    assert estimate.confidence == pytest.approx(0.0)
    assert estimate.has_min_observations is False


def test_blend_quality_confidence_scales_with_observations():
    returns = np.full(21, 0.01)

    estimate = blend_quality(
        0.2,
        returns,
        metric="log_growth",
        min_observations=20,
        full_weight_observations=63,
    )

    expected_raw_live = rolling_fitness(returns, metric="log_growth")
    expected_live = 0.20
    expected_confidence = 21 / 63

    assert estimate.raw_live_quality == pytest.approx(expected_raw_live)
    assert estimate.live_quality == pytest.approx(expected_live)
    assert estimate.confidence == pytest.approx(expected_confidence)
    assert estimate.blended_quality == pytest.approx(
        (1.0 - expected_confidence) * 0.2 + expected_confidence * expected_live
    )
    assert estimate.has_min_observations is True


def test_blend_quality_clips_and_shrinks_small_sample_live_quality():
    returns = np.array([0.0004958065767121531, 0.0005719862452852798], dtype=np.float64)

    estimate = blend_quality(
        0.0,
        returns,
        metric="sharpe",
        min_observations=20,
        full_weight_observations=63,
    )

    raw_live = rolling_fitness(returns, metric="sharpe")
    expected_effective = min(len(returns) / 20, 1.0) * 3.0
    expected_confidence = len(returns) / 63

    assert raw_live > 100.0
    assert estimate.raw_live_quality == pytest.approx(raw_live)
    assert estimate.live_quality == pytest.approx(expected_effective)
    assert estimate.confidence == pytest.approx(expected_confidence)
    assert estimate.blended_quality == pytest.approx(
        expected_confidence * expected_effective
    )


def test_rolling_fitness_accepts_numpy_arrays():
    returns = np.array([0.01, -0.005, 0.02], dtype=np.float64)

    fitness = rolling_fitness(returns, metric="log_growth")

    assert fitness == pytest.approx(np.mean(np.log1p(returns)) * 252)
