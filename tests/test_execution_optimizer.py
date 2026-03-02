"""Tests for ExecutionOptimizer."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from alpha_os.execution.optimizer import ExecutionConfig, ExecutionOptimizer


def _make_client(signals: dict[str, float | None]):
    """Create a mock SignalClient that returns predefined latest values."""
    client = MagicMock()

    def get_latest(name):
        if name not in signals:
            return None
        v = signals[name]
        if v is None:
            return None
        return {"timestamp": "2026-03-01T10:00:00Z", "value": v}

    client.get_latest = get_latest
    return client


def test_optimal_window_favorable():
    client = _make_client({
        "book_imbalance_btc": 0.05,
        "vpin_btc": 0.3,
        "spread_bps_btc": 2.0,
    })
    opt = ExecutionOptimizer(client)
    assert opt.optimal_execution_window("buy") is True
    assert opt.optimal_execution_window("sell") is True


def test_optimal_window_high_vpin():
    client = _make_client({
        "book_imbalance_btc": 0.05,
        "vpin_btc": 0.8,
        "spread_bps_btc": 2.0,
    })
    opt = ExecutionOptimizer(client)
    assert opt.optimal_execution_window("buy") is False


def test_optimal_window_wide_spread():
    client = _make_client({
        "book_imbalance_btc": 0.05,
        "vpin_btc": 0.3,
        "spread_bps_btc": 10.0,
    })
    opt = ExecutionOptimizer(client)
    assert opt.optimal_execution_window("buy") is False


def test_optimal_window_unfavorable_imbalance_buy():
    """Ask-heavy imbalance is bad for buyers."""
    client = _make_client({
        "book_imbalance_btc": -0.2,
        "vpin_btc": 0.3,
        "spread_bps_btc": 2.0,
    })
    opt = ExecutionOptimizer(client)
    assert opt.optimal_execution_window("buy") is False
    assert opt.optimal_execution_window("sell") is True


def test_optimal_window_unfavorable_imbalance_sell():
    """Bid-heavy imbalance is bad for sellers."""
    client = _make_client({
        "book_imbalance_btc": 0.2,
        "vpin_btc": 0.3,
        "spread_bps_btc": 2.0,
    })
    opt = ExecutionOptimizer(client)
    assert opt.optimal_execution_window("buy") is True
    assert opt.optimal_execution_window("sell") is False


def test_optimal_window_missing_signals():
    """Missing signals default to allow execution."""
    client = _make_client({})
    opt = ExecutionOptimizer(client)
    assert opt.optimal_execution_window("buy") is True


def test_split_order_good_conditions():
    client = _make_client({
        "vpin_btc": 0.2,
        "spread_bps_btc": 2.0,
    })
    opt = ExecutionOptimizer(client)
    slices = opt.split_order(1.0)
    assert slices == [1.0]


def test_split_order_high_vpin():
    client = _make_client({
        "vpin_btc": 0.8,
        "spread_bps_btc": 2.0,
    })
    opt = ExecutionOptimizer(client, ExecutionConfig(max_slices=5))
    slices = opt.split_order(1.0)
    assert len(slices) == 5
    assert sum(slices) == pytest.approx(1.0)


def test_split_order_wide_spread():
    client = _make_client({
        "vpin_btc": 0.3,
        "spread_bps_btc": 10.0,
    })
    opt = ExecutionOptimizer(client, ExecutionConfig(max_slices=5))
    slices = opt.split_order(1.0)
    assert len(slices) >= 3
    assert sum(slices) == pytest.approx(1.0)
