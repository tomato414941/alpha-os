from __future__ import annotations

import numpy as np
import pytest

from alpha_os.paper.simulator import _backfill_signals_to_position_intent


def test_consensus_backfill_matches_live_strategic_shape():
    signals = np.array([1.0, 0.8, -0.2])
    weights = np.array([0.5, 0.3, 0.2])

    raw, adjusted = _backfill_signals_to_position_intent(
        signals,
        weights,
        combine_mode="consensus",
        dd_scale=0.8,
        vol_scale=0.1,
    )

    mean = float(np.dot(weights, signals))
    std = float(np.sqrt(np.dot(weights, (signals - mean) ** 2)))
    consensus = abs(mean) / (abs(mean) + std)

    assert raw == pytest.approx(np.clip(mean, -1.0, 1.0))
    assert adjusted == pytest.approx(np.sign(mean) * consensus * 0.8)


def test_non_consensus_backfill_keeps_raw_times_risk_scales():
    signals = np.array([1.0, 0.8, -0.2])
    weights = np.array([0.5, 0.3, 0.2])

    raw, adjusted = _backfill_signals_to_position_intent(
        signals,
        weights,
        combine_mode="voting",
        dd_scale=0.8,
        vol_scale=0.5,
    )

    expected_raw = float(np.clip(np.dot(weights, signals), -1.0, 1.0))
    assert raw == pytest.approx(expected_raw)
    assert adjusted == pytest.approx(expected_raw * 0.8 * 0.5)
