"""Tests for the live CLI command and Trader rename."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_trader_alias():
    """PaperTrader alias still works after rename."""
    from alpha_os.paper.trader import Trader, PaperTrader
    assert PaperTrader is Trader


def test_executor_abc_has_defaults():
    """Executor ABC provides default implementations for optional methods."""
    from alpha_os.execution.executor import Executor, Order, Fill

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
