"""Tests for ExecutionOptimizer."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from alpha_os.execution.optimizer import ExecutionConfig, ExecutionOptimizer


def _make_client(signals: dict[str, float | None], *, latest_ts: str = "2026-03-01T10:00:00Z"):
    """Create a mock SignalClient that returns predefined latest values."""
    client = MagicMock()

    def get_latest(name):
        if name not in signals:
            return None
        v = signals[name]
        if v is None:
            return None
        return {"timestamp": latest_ts, "value": v, "name": name}

    client.get_latest = get_latest
    client.get_data.return_value = pd.DataFrame(columns=["timestamp", "value"])
    return client


def _fresh_ts() -> str:
    return datetime.now(UTC).isoformat()


def test_optimal_window_favorable():
    client = _make_client({
        "book_imbalance_btc": 0.05,
        "vpin_btc": 0.3,
        "spread_bps_btc": 2.0,
    }, latest_ts=_fresh_ts())
    opt = ExecutionOptimizer(client)
    assert opt.optimal_execution_window("buy") is True
    assert opt.optimal_execution_window("sell") is True


def test_optimal_window_high_vpin():
    client = _make_client({
        "book_imbalance_btc": 0.05,
        "vpin_btc": 0.8,
        "spread_bps_btc": 2.0,
    }, latest_ts=_fresh_ts())
    opt = ExecutionOptimizer(client, ExecutionConfig(vpin_threshold=0.5))
    assert opt.optimal_execution_window("buy") is False


def test_optimal_window_wide_spread():
    client = _make_client({
        "book_imbalance_btc": 0.05,
        "vpin_btc": 0.3,
        "spread_bps_btc": 10.0,
    }, latest_ts=_fresh_ts())
    opt = ExecutionOptimizer(client)
    assert opt.optimal_execution_window("buy") is False


def test_optimal_window_unfavorable_imbalance_buy():
    """Ask-heavy imbalance is bad for buyers."""
    client = _make_client({
        "book_imbalance_btc": -0.2,
        "vpin_btc": 0.3,
        "spread_bps_btc": 2.0,
    }, latest_ts=_fresh_ts())
    opt = ExecutionOptimizer(client, ExecutionConfig(imbalance_threshold=0.1))
    assert opt.optimal_execution_window("buy") is False
    assert opt.optimal_execution_window("sell") is True


def test_optimal_window_unfavorable_imbalance_sell():
    """Bid-heavy imbalance is bad for sellers."""
    client = _make_client({
        "book_imbalance_btc": 0.2,
        "vpin_btc": 0.3,
        "spread_bps_btc": 2.0,
    }, latest_ts=_fresh_ts())
    opt = ExecutionOptimizer(client, ExecutionConfig(imbalance_threshold=0.1))
    assert opt.optimal_execution_window("buy") is True
    assert opt.optimal_execution_window("sell") is False


def test_optimal_window_missing_signals():
    """Missing signals default to allow execution."""
    client = _make_client({})
    opt = ExecutionOptimizer(client)
    assert opt.optimal_execution_window("buy") is True


def test_optimal_window_prefers_recent_signal_data():
    client = _make_client(
        {
            "book_imbalance_btc": 0.05,
            "vpin_btc": 0.95,
            "spread_bps_btc": 2.0,
        },
        latest_ts="2026-03-08T00:00:00Z",
    )
    client.get_data.side_effect = lambda name, since=None: pd.DataFrame(
        [{"timestamp": pd.Timestamp("2026-03-08T14:00:00Z"), "value": {
            "book_imbalance_btc": 0.05,
            "vpin_btc": 0.30,
            "spread_bps_btc": 2.0,
        }[name]}]
    )
    opt = ExecutionOptimizer(client, ExecutionConfig(max_signal_age_seconds=300))
    assert opt.optimal_execution_window("buy") is True


def test_stale_latest_signal_does_not_block_when_recent_missing():
    client = _make_client(
        {
            "book_imbalance_btc": 0.05,
            "vpin_btc": 0.95,
            "spread_bps_btc": 2.0,
        },
        latest_ts="2026-03-08T00:00:00Z",
    )
    opt = ExecutionOptimizer(client, ExecutionConfig(max_signal_age_seconds=1))
    assert opt.optimal_execution_window("buy") is True


def test_split_order_good_conditions():
    client = _make_client({
        "vpin_btc": 0.2,
        "spread_bps_btc": 2.0,
    }, latest_ts=_fresh_ts())
    opt = ExecutionOptimizer(client)
    slices = opt.split_order(1.0)
    assert slices == [1.0]


def test_split_order_high_vpin():
    client = _make_client({
        "vpin_btc": 0.8,
        "spread_bps_btc": 2.0,
    }, latest_ts=_fresh_ts())
    opt = ExecutionOptimizer(client, ExecutionConfig(max_slices=5, vpin_threshold=0.5))
    slices = opt.split_order(1.0)
    assert len(slices) == 5
    assert sum(slices) == pytest.approx(1.0)


def test_split_order_wide_spread():
    client = _make_client({
        "vpin_btc": 0.3,
        "spread_bps_btc": 10.0,
    }, latest_ts=_fresh_ts())
    opt = ExecutionOptimizer(client, ExecutionConfig(max_slices=5))
    slices = opt.split_order(1.0)
    assert len(slices) >= 3
    assert sum(slices) == pytest.approx(1.0)
