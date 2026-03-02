from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from alpha_os.risk.distributional import (
    estimate_distribution,
    kelly_scale,
    passes_distributional_gate,
)
from alpha_os.risk.manager import RiskManager


def test_estimate_distribution_not_ready():
    returns = np.array([0.01, -0.02, 0.03])
    stats = estimate_distribution(returns, window=63, min_samples=10)
    assert not stats.ready
    assert stats.sample_size == 3


def test_estimate_distribution_ready_and_cvar():
    returns = np.array([-0.05, -0.03, -0.02, 0.01, 0.02, 0.03, 0.04, 0.01])
    stats = estimate_distribution(returns, window=0, min_samples=5, cvar_alpha=0.25)
    assert stats.ready
    assert stats.sample_size == 8
    assert stats.cvar < 0.0


def test_distributional_gate_blocks_on_tail_risk():
    stats = estimate_distribution(
        np.array([-0.06, -0.05, -0.04, -0.03, 0.01, 0.01, 0.01, 0.01]),
        window=0,
        min_samples=5,
        cvar_alpha=0.25,
    )
    assert not passes_distributional_gate(
        stats,
        max_left_tail_prob=1.0,
        max_cvar_abs=0.02,
    )


def test_kelly_scale_clipped_between_zero_and_one():
    good = estimate_distribution(
        np.array([0.01, 0.02, 0.015, 0.01, 0.005, 0.02]),
        window=0,
        min_samples=5,
    )
    bad = estimate_distribution(
        np.array([-0.01, -0.02, -0.015, -0.01, -0.005, -0.02]),
        window=0,
        min_samples=5,
    )
    assert 0.0 <= kelly_scale(good, kelly_fraction=0.5, max_leverage=1.0) <= 1.0
    assert kelly_scale(bad, kelly_fraction=0.5, max_leverage=1.0) == 0.0


def test_risk_manager_distributional_adjustment():
    rm = RiskManager()
    cfg = SimpleNamespace(
        window=63,
        min_samples=20,
        tail_sigma=2.0,
        cvar_alpha=0.05,
        max_left_tail_prob=0.20,
        max_cvar_abs=0.05,
        kelly_fraction=0.5,
        max_kelly_leverage=1.0,
    )
    rng = np.random.RandomState(42)
    returns = rng.normal(0.001, 0.01, 100)
    gate_ok, scale, stats = rm.distributional_adjustment(returns, cfg)
    assert gate_ok
    assert stats.ready
    assert 0.0 <= scale <= 1.0
