from __future__ import annotations

import numpy as np
import pytest

from alpha_os.alpha.registry import AlphaRecord, AlphaState
from alpha_os.alpha.lifecycle import ST_ACTIVE, ST_DORMANT, ST_PROBATION
from alpha_os.paper.simulator import (
    ST_EXCLUDED,
    _backfill_signals_to_position_intent,
    _initial_simulation_state,
    _live_like_eval_indices,
)


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
    assert adjusted == pytest.approx(expected_raw * 0.8)


def test_raw_mean_sizing_mode_ignores_consensus_and_vol_scale():
    signals = np.array([1.0, 0.8, -0.2])
    weights = np.array([0.5, 0.3, 0.2])

    raw, adjusted = _backfill_signals_to_position_intent(
        signals,
        weights,
        combine_mode="consensus",
        dd_scale=0.8,
        vol_scale=0.1,
        sizing_mode="raw_mean",
    )

    expected_raw = float(np.clip(np.dot(weights, signals), -1.0, 1.0))
    assert raw == pytest.approx(expected_raw)
    assert adjusted == pytest.approx(expected_raw * 0.8)


def test_initial_simulation_state_preserves_registry_state():
    assert _initial_simulation_state(AlphaState.ACTIVE) == ST_ACTIVE
    assert _initial_simulation_state(AlphaState.PROBATION) == ST_PROBATION
    assert _initial_simulation_state(AlphaState.DORMANT) == ST_DORMANT
    assert _initial_simulation_state(AlphaState.REJECTED) == ST_EXCLUDED
    assert _initial_simulation_state(AlphaState.BORN) == ST_EXCLUDED


def test_live_like_eval_indices_matches_live_candidate_rules():
    records = [
        AlphaRecord(alpha_id="a0", expression="x", state=AlphaState.ACTIVE, oos_sharpe=0.9),
        AlphaRecord(alpha_id="a1", expression="x", state=AlphaState.PROBATION, oos_sharpe=0.8),
        AlphaRecord(alpha_id="a2", expression="x", state=AlphaState.ACTIVE, oos_sharpe=0.7),
        AlphaRecord(alpha_id="a3", expression="x", state=AlphaState.ACTIVE, oos_sharpe=0.6),
        AlphaRecord(alpha_id="a4", expression="x", state=AlphaState.ACTIVE, oos_sharpe=0.5),
        AlphaRecord(alpha_id="a5", expression="x", state=AlphaState.ACTIVE, oos_sharpe=0.4),
        AlphaRecord(alpha_id="d0", expression="x", state=AlphaState.DORMANT, oos_sharpe=9.9),
        AlphaRecord(alpha_id="r0", expression="x", state=AlphaState.REJECTED, oos_sharpe=9.8),
    ]
    state_codes = np.array([
        ST_ACTIVE,
        ST_PROBATION,
        ST_ACTIVE,
        ST_ACTIVE,
        ST_ACTIVE,
        ST_ACTIVE,
        ST_DORMANT,
        ST_EXCLUDED,
    ])

    trading_candidates, dormant, eval_set = _live_like_eval_indices(
        records,
        state_codes,
        max_trading=1,
        metric="sharpe",
    )

    assert trading_candidates == [0, 1, 2, 3, 4]
    assert dormant == [6]
    assert eval_set == [0, 1, 2, 3, 4, 6]
