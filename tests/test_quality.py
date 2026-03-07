from __future__ import annotations

import numpy as np
import pytest

from alpha_os.alpha.quality import (
    blend_quality,
    confidence_weight_scale,
    rolling_fitness,
    shrink_weight_quality,
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

    expected_live = rolling_fitness(returns, metric="log_growth")
    expected_confidence = 21 / 63

    assert estimate.live_quality == pytest.approx(expected_live)
    assert estimate.confidence == pytest.approx(expected_confidence)
    assert estimate.blended_quality == pytest.approx(
        (1.0 - expected_confidence) * 0.2 + expected_confidence * expected_live
    )
    assert estimate.has_min_observations is True


def test_rolling_fitness_accepts_numpy_arrays():
    returns = np.array([0.01, -0.005, 0.02], dtype=np.float64)

    fitness = rolling_fitness(returns, metric="log_growth")

    assert fitness == pytest.approx(np.mean(np.log1p(returns)) * 252)


def test_confidence_weight_scale_respects_floor_and_power():
    confidence = np.array([0.0, 0.25, 1.0], dtype=np.float64)

    scale = confidence_weight_scale(confidence, floor=0.2, power=2.0)

    assert scale[0] == pytest.approx(0.2)
    assert scale[1] == pytest.approx(0.2 + 0.8 * 0.25**2)
    assert scale[2] == pytest.approx(1.0)


def test_shrink_weight_quality_reduces_low_confidence_scores():
    shrunk = shrink_weight_quality(
        np.array([0.8, 0.8], dtype=np.float64),
        np.array([0.1, 1.0], dtype=np.float64),
        floor=0.0,
        power=2.0,
    )

    assert shrunk[0] == pytest.approx(0.8 * 0.1**2)
    assert shrunk[1] == pytest.approx(0.8)
