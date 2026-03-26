from __future__ import annotations

import numpy as np
import pytest

from alpha_os.hypotheses.state_lifecycle import ST_ACTIVE, ST_CANDIDATE, ST_DORMANT
from alpha_os.legacy.managed_alphas import AlphaRecord, AlphaState
from alpha_os.config import Config
from alpha_os.legacy.replay_simulator import (
    ST_EXCLUDED,
    _apply_regime_adjustment,
    _replay_signals_to_position_intent,
    _initial_simulation_state,
    _live_like_eval_indices,
)


def test_consensus_replay_matches_runtime_strategic_shape():
    signals = np.array([1.0, 0.8, -0.2])
    weights = np.array([0.5, 0.3, 0.2])

    raw, adjusted = _replay_signals_to_position_intent(
        signals,
        weights,
        dd_scale=0.8,
        vol_scale=0.1,
    )

    mean = float(np.dot(weights, signals))
    std = float(np.sqrt(np.dot(weights, (signals - mean) ** 2)))
    consensus = abs(mean) / (abs(mean) + std)

    assert raw == pytest.approx(np.clip(mean, -1.0, 1.0))
    assert adjusted == pytest.approx(np.sign(mean) * consensus * 0.8)


def test_raw_mean_sizing_mode_ignores_consensus():
    signals = np.array([1.0, 0.8, -0.2])
    weights = np.array([0.5, 0.3, 0.2])

    raw, adjusted = _replay_signals_to_position_intent(
        signals,
        weights,
        dd_scale=0.8,
        vol_scale=0.1,
        sizing_mode="raw_mean",
    )

    expected_raw = float(np.clip(np.dot(weights, signals), -1.0, 1.0))
    assert raw == pytest.approx(expected_raw)
    assert adjusted == pytest.approx(expected_raw * 0.8)


def test_initial_simulation_state_preserves_registry_state():
    assert _initial_simulation_state(AlphaState.ACTIVE) == ST_ACTIVE
    assert _initial_simulation_state(AlphaState.CANDIDATE) == ST_EXCLUDED
    assert _initial_simulation_state(AlphaState.DORMANT) == ST_DORMANT
    assert _initial_simulation_state(AlphaState.REJECTED) == ST_EXCLUDED


def test_live_like_eval_indices_matches_live_candidate_rules():
    records = [
        AlphaRecord(alpha_id="a0", expression="x", state=AlphaState.ACTIVE, oos_sharpe=0.9),
        AlphaRecord(alpha_id="a1", expression="x", state=AlphaState.CANDIDATE, oos_sharpe=0.8),
        AlphaRecord(alpha_id="a2", expression="x", state=AlphaState.ACTIVE, oos_sharpe=0.7),
        AlphaRecord(alpha_id="a3", expression="x", state=AlphaState.ACTIVE, oos_sharpe=0.6),
        AlphaRecord(alpha_id="a4", expression="x", state=AlphaState.ACTIVE, oos_sharpe=0.5),
        AlphaRecord(alpha_id="a5", expression="x", state=AlphaState.ACTIVE, oos_sharpe=0.4),
        AlphaRecord(alpha_id="d0", expression="x", state=AlphaState.DORMANT, oos_sharpe=9.9),
        AlphaRecord(alpha_id="r0", expression="x", state=AlphaState.REJECTED, oos_sharpe=9.8),
    ]
    state_codes = np.array([
        ST_ACTIVE,
        ST_CANDIDATE,
        ST_ACTIVE,
        ST_ACTIVE,
        ST_ACTIVE,
        ST_ACTIVE,
        ST_DORMANT,
        ST_EXCLUDED,
    ])
    prior_quality = np.array([r.oos_sharpe for r in records], dtype=np.float64)
    blended_quality = np.array([0.2, 0.7, 0.8, 0.6, 0.95, 0.1, 0.5, 0.0], dtype=np.float64)
    confidence = np.array([0.3, 0.2, 0.4, 0.6, 0.1, 0.9, 0.0, 0.0], dtype=np.float64)

    trading_candidates, dormant, eval_set = _live_like_eval_indices(
        records,
        state_codes,
        prior_quality=prior_quality,
        blended_quality=blended_quality,
        confidence=confidence,
        max_trading=1,
        metric="sharpe",
        shortlist_preselect_factor=10,
    )

    assert trading_candidates == [4, 2, 3, 0, 5]
    assert dormant == [6]
    assert eval_set == [4, 2, 3, 0, 5, 6]


def test_apply_regime_adjustment_is_noop_when_not_triggered(monkeypatch):
    cfg = Config.load()
    cfg.regime.enabled = True
    cfg.regime.long_window = 3
    cfg.regime.short_window = 2
    cfg.regime.drift_threshold = 0.3

    class DummyDetector:
        def __init__(self, short_window, long_window):
            assert short_window == 2
            assert long_window == 3

        def detect(self, returns):
            return type("Regime", (), {"drift_score": 0.2})()

    monkeypatch.setattr("alpha_os.legacy.replay_simulator.RegimeDetector", DummyDetector)

    adjusted = _apply_regime_adjustment(
        0.6,
        np.array([0.01, -0.01, 0.02]),
        cfg,
    )
    assert adjusted == pytest.approx(0.6)


def test_apply_regime_adjustment_scales_signal_when_triggered(monkeypatch):
    cfg = Config.load()
    cfg.regime.enabled = True
    cfg.regime.long_window = 3
    cfg.regime.short_window = 2
    cfg.regime.drift_threshold = 0.3
    cfg.regime.drift_position_scale_min = 0.5

    class DummyDetector:
        def __init__(self, short_window, long_window):
            pass

        def detect(self, returns):
            return type("Regime", (), {"drift_score": 0.7})()

    monkeypatch.setattr("alpha_os.legacy.replay_simulator.RegimeDetector", DummyDetector)

    adjusted = _apply_regime_adjustment(
        0.6,
        np.array([0.01, -0.01, 0.02]),
        cfg,
    )
    assert adjusted == pytest.approx(0.3)
