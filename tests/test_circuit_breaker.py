"""Tests for CircuitBreaker — all safety limits, persistence, and daily reset."""
from __future__ import annotations


import pytest

from alpha_os.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)


@pytest.fixture
def cb(tmp_path):
    """Fresh circuit breaker with state file in tmp dir."""
    config = CircuitBreakerConfig(
        daily_loss_limit_pct=3.0,
        max_consecutive_losses=5,
        max_drawdown_pct=10.0,
        kill_file=str(tmp_path / "KILL_SWITCH"),
    )
    return CircuitBreaker(config=config)


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "cb_state.json"


# ---------------------------------------------------------------------------
# Basic safety check
# ---------------------------------------------------------------------------

def test_safe_by_default(cb):
    safe, reason = cb.is_safe_to_trade(10000.0)
    assert safe is True
    assert reason == "ok"


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

def test_kill_switch(cb, tmp_path):
    kill_path = tmp_path / "KILL_SWITCH"
    kill_path.touch()
    safe, reason = cb.is_safe_to_trade(10000.0)
    assert safe is False
    assert "Kill switch" in reason


# ---------------------------------------------------------------------------
# Daily loss limit
# ---------------------------------------------------------------------------

def test_daily_loss_limit(cb):
    cb._daily_start_equity = 10000.0
    # Record $350 loss (3.5% > 3% limit)
    cb.record_trade(-350.0)
    safe, reason = cb.is_safe_to_trade(9650.0)
    assert safe is False
    assert "Daily loss" in reason
    assert cb.halted is True


def test_daily_loss_under_limit(cb):
    cb._daily_start_equity = 10000.0
    cb.record_trade(-200.0)  # 2% < 3%
    safe, reason = cb.is_safe_to_trade(9800.0)
    assert safe is True


# ---------------------------------------------------------------------------
# Consecutive losses
# ---------------------------------------------------------------------------

def test_consecutive_losses(cb):
    cb._daily_start_equity = 10000.0
    for _ in range(5):
        cb.record_trade(-10.0)
    safe, reason = cb.is_safe_to_trade(9950.0)
    assert safe is False
    assert "Consecutive losses" in reason


def test_consecutive_losses_reset_on_win(cb):
    cb._daily_start_equity = 10000.0
    for _ in range(4):
        cb.record_trade(-10.0)
    cb.record_trade(50.0)  # Win resets counter
    safe, _ = cb.is_safe_to_trade(10010.0)
    assert safe is True


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown(cb):
    cb._peak_equity = 10000.0
    safe, reason = cb.is_safe_to_trade(8900.0)  # 11% DD > 10% limit
    assert safe is False
    assert "Drawdown" in reason


def test_drawdown_under_limit(cb):
    cb._peak_equity = 10000.0
    safe, _ = cb.is_safe_to_trade(9200.0)  # 8% < 10%
    assert safe is True


def test_peak_equity_updates(cb):
    cb.is_safe_to_trade(10000.0)
    assert cb._peak_equity == 10000.0
    cb.is_safe_to_trade(11000.0)
    assert cb._peak_equity == 11000.0
    cb.is_safe_to_trade(10500.0)
    assert cb._peak_equity == 11000.0  # Peak doesn't decrease


# ---------------------------------------------------------------------------
# Halted state persists
# ---------------------------------------------------------------------------

def test_halted_stays_halted(cb):
    cb._halted = True
    cb._halt_reason = "test halt"
    safe, reason = cb.is_safe_to_trade(10000.0)
    assert safe is False
    assert "test halt" in reason


# ---------------------------------------------------------------------------
# Persistence (save/load)
# ---------------------------------------------------------------------------

def test_save_and_load(cb, state_path):
    cb._daily_start_equity = 10000.0
    cb._peak_equity = 12000.0
    cb.record_trade(-100.0)
    cb.record_trade(-50.0)
    cb.save(state_path)

    loaded = CircuitBreaker.load(config=cb.config, path=state_path)
    assert loaded._daily_pnl == -150.0
    assert loaded._consecutive_losses == 2
    assert loaded._peak_equity == 12000.0


def test_load_missing_file(state_path):
    loaded = CircuitBreaker.load(path=state_path)
    assert loaded._daily_pnl == 0.0
    assert loaded.halted is False


def test_load_corrupted_file(state_path):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("not valid json{{{")
    loaded = CircuitBreaker.load(path=state_path)
    assert loaded.halted is False


# ---------------------------------------------------------------------------
# Daily reset
# ---------------------------------------------------------------------------

def test_daily_reset_clears_halt(cb, state_path, monkeypatch):
    cb._halted = True
    cb._halt_reason = "old halt"
    cb._daily_pnl = -500.0
    cb._current_date = "2026-02-26"

    # Simulate next day
    from datetime import date as date_cls
    monkeypatch.setattr(
        "alpha_os.risk.circuit_breaker.date",
        type("MockDate", (), {"today": staticmethod(lambda: date_cls(2026, 2, 27))}),
    )
    cb.save = lambda path=None: None  # Skip file write in test
    cb.reset_daily(9500.0)

    assert cb.halted is False
    assert cb._halt_reason == ""
    assert cb._daily_pnl == 0.0
    assert cb._daily_start_equity == 9500.0


def test_daily_reset_noop_same_day(cb, monkeypatch):
    from datetime import date as date_cls
    monkeypatch.setattr(
        "alpha_os.risk.circuit_breaker.date",
        type("MockDate", (), {"today": staticmethod(lambda: date_cls(2026, 2, 27))}),
    )
    cb._current_date = "2026-02-27"
    cb._daily_pnl = -100.0
    cb.save = lambda path=None: None
    cb.reset_daily(10000.0)
    assert cb._daily_pnl == -100.0  # Not reset — same day


# ---------------------------------------------------------------------------
# Check order: kill switch > halted > daily loss > consec > drawdown
# ---------------------------------------------------------------------------

def test_check_order_kill_switch_first(cb, tmp_path):
    """Kill switch takes priority even when other limits are also breached."""
    kill_path = tmp_path / "KILL_SWITCH"
    kill_path.touch()
    cb._halted = True
    cb._halt_reason = "daily loss"
    safe, reason = cb.is_safe_to_trade(10000.0)
    assert safe is False
    assert "Kill switch" in reason  # Not "daily loss"
