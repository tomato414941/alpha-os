"""Tests for the live CLI command, Trader rename, and Phase 4 features."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_trader_alias():
    """PaperTrader alias still works after rename."""
    from alpha_os.paper.trader import Trader, PaperTrader
    assert PaperTrader is Trader


def test_executor_abc_has_defaults():
    """Executor ABC provides default implementations for optional methods."""
    from alpha_os.execution.executor import Executor

    class MinimalExecutor(Executor):
        def submit_order(self, order):
            return None

        def get_position(self, symbol):
            return 0.0

        def get_cash(self):
            return 1000.0

    ex = MinimalExecutor()
    ex.set_price("BTC", 50000.0)  # no-op, should not raise
    assert ex.portfolio_value == 1000.0  # defaults to get_cash()
    assert ex.all_positions == {}
    assert ex.all_fills == []


def test_binance_executor_portfolio_value():
    """BinanceExecutor.portfolio_value sums USDT + marked positions."""
    from alpha_os.execution.binance import BinanceExecutor

    mock_ex = MagicMock()
    mock_ex.fetch_balance.return_value = {
        "USDT": {"total": 5000.0, "free": 5000.0},
        "BTC": {"total": 0.1, "free": 0.1},
        "free": {}, "used": {}, "total": {}, "info": {},
    }
    mock_ex.fetch_ticker.return_value = {"last": 50000.0}

    executor = object.__new__(BinanceExecutor)
    executor._exchange = mock_ex
    executor._testnet = True
    executor._symbol_map = {}
    executor._max_slippage_bps = 10.0
    executor._max_book_fraction = 0.1

    assert executor.portfolio_value == pytest.approx(10000.0)  # 5000 + 0.1*50000


def test_binance_executor_all_positions():
    """BinanceExecutor.all_positions returns non-zero holdings."""
    from alpha_os.execution.binance import BinanceExecutor

    mock_ex = MagicMock()
    mock_ex.fetch_balance.return_value = {
        "USDT": {"total": 5000.0},
        "BTC": {"total": 0.1},
        "ETH": {"total": 0.0},
        "free": {}, "used": {}, "total": {}, "info": {},
    }

    executor = object.__new__(BinanceExecutor)
    executor._exchange = mock_ex
    executor._testnet = True
    executor._symbol_map = {}
    executor._max_slippage_bps = 10.0
    executor._max_book_fraction = 0.1

    positions = executor.all_positions
    assert "BTC" in positions
    assert positions["BTC"] == 0.1
    assert "ETH" not in positions  # zero position filtered
    assert "USDT" not in positions  # USDT excluded


def test_live_parser():
    """CLI parser accepts live subcommand with correct defaults."""
    from alpha_os.cli import _build_parser

    parser = _build_parser()

    # Testnet (default)
    args = parser.parse_args(["live", "--once"])
    assert args.command == "live"
    assert args.once is True
    assert args.real is False
    assert args.capital == 10000.0
    assert args.asset == "BTC"

    # Real mode
    args = parser.parse_args(["live", "--once", "--real", "--capital", "500"])
    assert args.real is True
    assert args.capital == 500.0


# ---------------------------------------------------------------------------
# Position display filtering
# ---------------------------------------------------------------------------

def test_print_status_filters_positions(capsys):
    """print_status shows only the traded asset, hides others."""
    from alpha_os.paper.trader import Trader
    from alpha_os.paper.tracker import PaperTradingSummary

    trader = object.__new__(Trader)
    trader.price_signal = "btc_ohlcv"

    summary = PaperTradingSummary(
        start_date="2026-02-20",
        end_date="2026-02-27",
        n_days=7,
        initial_value=10000.0,
        final_value=10500.0,
        total_return=0.05,
        sharpe=1.5,
        max_drawdown=0.02,
        total_trades=5,
        current_positions={
            "BTC": 0.1,
            "ETH": 1.0,
            "SOL": 5.0,
            "DOGE": 10000.0,
        },
        current_cash=5000.0,
    )
    trader.portfolio_tracker = MagicMock()
    trader.portfolio_tracker.summary.return_value = summary

    trader.print_status()
    output = capsys.readouterr().out

    assert "BTC: 0.100000" in output
    assert "ETH" not in output
    assert "SOL" not in output
    assert "DOGE" not in output
    assert "3 other positions hidden" in output


def test_print_status_no_positions(capsys):
    """print_status handles empty positions."""
    from alpha_os.paper.trader import Trader
    from alpha_os.paper.tracker import PaperTradingSummary

    trader = object.__new__(Trader)
    trader.price_signal = "btc_ohlcv"

    summary = PaperTradingSummary(
        start_date="2026-02-27",
        end_date="2026-02-27",
        n_days=1,
        initial_value=10000.0,
        final_value=10000.0,
        total_return=0.0,
        sharpe=0.0,
        max_drawdown=0.0,
        total_trades=0,
        current_positions={},
        current_cash=10000.0,
    )
    trader.portfolio_tracker = MagicMock()
    trader.portfolio_tracker.summary.return_value = summary

    trader.print_status()
    output = capsys.readouterr().out

    assert "hidden" not in output


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def test_reconcile_match():
    """reconcile() reports match when internal == exchange."""
    from alpha_os.paper.trader import Trader
    from alpha_os.paper.tracker import PortfolioSnapshot

    trader = object.__new__(Trader)
    trader.price_signal = "btc_ohlcv"

    snapshot = PortfolioSnapshot(
        date="2026-02-27",
        cash=5000.0,
        positions={"BTC": 0.1},
        portfolio_value=10000.0,
        daily_pnl=100.0,
        daily_return=0.01,
        combined_signal=0.5,
        dd_scale=1.0,
        vol_scale=1.0,
    )
    trader.portfolio_tracker = MagicMock()
    trader.portfolio_tracker.get_last_snapshot.return_value = snapshot

    trader.executor = MagicMock()
    trader.executor.get_position.return_value = 0.1
    trader.executor.get_cash.return_value = 5000.0

    result = trader.reconcile()
    assert result["match"] is True
    assert result["qty_diff"] < 1e-6
    assert result["cash_diff"] < 1.0


def test_reconcile_mismatch():
    """reconcile() detects position mismatch."""
    from alpha_os.paper.trader import Trader
    from alpha_os.paper.tracker import PortfolioSnapshot

    trader = object.__new__(Trader)
    trader.price_signal = "btc_ohlcv"

    snapshot = PortfolioSnapshot(
        date="2026-02-27",
        cash=5000.0,
        positions={"BTC": 0.1},
        portfolio_value=10000.0,
        daily_pnl=100.0,
        daily_return=0.01,
        combined_signal=0.5,
        dd_scale=1.0,
        vol_scale=1.0,
    )
    trader.portfolio_tracker = MagicMock()
    trader.portfolio_tracker.get_last_snapshot.return_value = snapshot

    trader.executor = MagicMock()
    trader.executor.get_position.return_value = 0.15  # Mismatch
    trader.executor.get_cash.return_value = 4800.0  # Mismatch

    result = trader.reconcile()
    assert result["match"] is False
    assert result["qty_diff"] == pytest.approx(0.05)
    assert result["cash_diff"] == pytest.approx(200.0)


def test_reconcile_no_data():
    """reconcile() returns no_data when no snapshots exist."""
    from alpha_os.paper.trader import Trader

    trader = object.__new__(Trader)
    trader.price_signal = "btc_ohlcv"

    trader.portfolio_tracker = MagicMock()
    trader.portfolio_tracker.get_last_snapshot.return_value = None

    result = trader.reconcile()
    assert result["status"] == "no_data"
