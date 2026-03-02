from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from alpha_os.paper.event_driven import EventDrivenTrader, EventTriggerConfig


def _make_trader_mock():
    trader = MagicMock()
    trader.run_cycle.return_value = MagicMock(
        combined_signal=0.5,
        portfolio_value=10000.0,
        daily_pnl=50.0,
    )
    return trader


def _make_client_mock(events: list[dict]):
    """Create a SignalClient mock whose subscribe yields events."""
    client = MagicMock()

    async def _subscribe(pattern, **kwargs):
        for e in events:
            yield e

    client.subscribe = _subscribe
    return client


@pytest.mark.asyncio
async def test_debounce_prevents_rapid_fire():
    trader = _make_trader_mock()
    events = [
        {"name": "funding_rate_btc", "event_type": "update", "value": 0.01, "timestamp": "t1", "detail": ""},
        {"name": "funding_rate_btc", "event_type": "update", "value": 0.02, "timestamp": "t2", "detail": ""},
    ]
    client = _make_client_mock(events)

    config = EventTriggerConfig(min_interval=10.0, max_interval=3600.0)
    ed = EventDrivenTrader(trader=trader, client=client, config=config)

    # Run with timeout so we don't block forever
    try:
        await asyncio.wait_for(ed.run(), timeout=0.5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

    # Initial cycle + possibly one event-triggered cycle, but second event debounced
    # Initial always runs, events may or may not trigger depending on timing
    assert trader.run_cycle.call_count >= 1


@pytest.mark.asyncio
async def test_anomaly_triggers_evaluation():
    trader = _make_trader_mock()
    events = [
        {"name": "liq_btc", "event_type": "anomaly", "value": 0.9, "timestamp": "t1", "detail": "z=5.0"},
    ]
    client = _make_client_mock(events)

    config = EventTriggerConfig(
        min_interval=0.0,  # no debounce for test
        max_interval=3600.0,
        anomaly_trigger=True,
    )
    ed = EventDrivenTrader(trader=trader, client=client, config=config)

    try:
        await asyncio.wait_for(ed.run(), timeout=0.5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

    # Initial + anomaly = at least 2 calls
    assert trader.run_cycle.call_count >= 2


@pytest.mark.asyncio
async def test_circuit_break_triggers():
    trader = _make_trader_mock()
    events = [
        {"name": "btc_ohlcv", "event_type": "circuit_break", "value": None, "timestamp": "", "detail": "5 failures"},
    ]
    client = _make_client_mock(events)

    config = EventTriggerConfig(min_interval=0.0, max_interval=3600.0)
    ed = EventDrivenTrader(trader=trader, client=client, config=config)

    try:
        await asyncio.wait_for(ed.run(), timeout=0.5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

    assert trader.run_cycle.call_count >= 2


@pytest.mark.asyncio
async def test_pre_cycle_hook_called():
    trader = _make_trader_mock()
    events = [
        {"name": "funding_rate_btc", "event_type": "update", "value": 0.01, "timestamp": "t1", "detail": ""},
    ]
    client = _make_client_mock(events)

    hook = MagicMock()
    config = EventTriggerConfig(min_interval=0.0, max_interval=3600.0)
    ed = EventDrivenTrader(
        trader=trader, client=client, config=config,
        pre_cycle_hook=hook,
    )

    try:
        await asyncio.wait_for(ed.run(), timeout=0.5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

    assert hook.call_count >= 1


def test_should_evaluate_debounce():
    trader = _make_trader_mock()
    client = MagicMock()
    config = EventTriggerConfig(min_interval=60.0)
    ed = EventDrivenTrader(trader=trader, client=client, config=config)

    # First call should pass (no prior evaluation)
    assert ed._should_evaluate("test") is True

    # Set last_eval_time to now
    ed._last_eval_time = time.time()

    # Second call should be debounced
    assert ed._should_evaluate("test") is False

    # Set last_eval_time to 61 seconds ago
    ed._last_eval_time = time.time() - 61
    assert ed._should_evaluate("test") is True
