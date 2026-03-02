"""Tests for BinanceExecutor — all exchange calls are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock


from alpha_os.execution.binance import (
    BinanceExecutor,
    _get_credentials,
    _load_secrets,
)
from alpha_os.execution.executor import Order


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def test_load_secrets_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.execution.binance.Path.home", lambda: tmp_path)
    assert _load_secrets() == {}


def test_load_secrets_parses_file(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.execution.binance.Path.home", lambda: tmp_path)
    secrets_dir = tmp_path / ".secrets"
    secrets_dir.mkdir()
    (secrets_dir / "binance").write_text(
        "export BINANCE_API_KEY='abc123'\nBINANCE_SECRET_KEY=secret456\n"
    )
    result = _load_secrets()
    assert result == {"BINANCE_API_KEY": "abc123", "BINANCE_SECRET_KEY": "secret456"}


def test_get_credentials_from_env(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "envkey")
    monkeypatch.setenv("BINANCE_SECRET_KEY", "envsecret")
    key, secret = _get_credentials()
    assert key == "envkey"
    assert secret == "envsecret"


# ---------------------------------------------------------------------------
# Helpers to build a mock exchange
# ---------------------------------------------------------------------------

def _mock_exchange():
    ex = MagicMock()
    ex.fetch_balance.return_value = {
        "BTC": {"total": 1.5, "free": 1.5},
        "USDT": {"total": 10000.0, "free": 10000.0},
    }
    ex.fetch_order_book.return_value = {
        "asks": [[50000.0, 2.0], [50010.0, 3.0], [50020.0, 5.0]],
        "bids": [[49990.0, 2.0], [49980.0, 3.0], [49970.0, 5.0]],
    }
    ex.amount_to_precision.side_effect = lambda sym, qty: str(round(qty, 8))
    ex.create_market_buy_order.return_value = {
        "id": "buy-001",
        "average": 50002.0,
        "filled": 0.1,
        "cost": 5000.2,
    }
    ex.create_market_sell_order.return_value = {
        "id": "sell-001",
        "average": 49988.0,
        "filled": 0.5,
        "cost": 24994.0,
    }
    return ex


def _make_executor(exchange=None):
    """Create BinanceExecutor with a mock exchange, bypassing create_spot_exchange."""
    executor = object.__new__(BinanceExecutor)
    executor._exchange = exchange or _mock_exchange()
    executor._testnet = True
    executor._symbol_map = {}
    executor._max_slippage_bps = 10.0
    executor._max_book_fraction = 0.1
    executor._optimizer = None
    return executor


# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------

def test_market_symbol_default():
    ex = _make_executor()
    assert ex._market_symbol("BTC") == "BTC/USDT"


def test_market_symbol_passthrough():
    ex = _make_executor()
    assert ex._market_symbol("BTC/USDT") == "BTC/USDT"


def test_market_symbol_custom_map():
    ex = _make_executor()
    ex._symbol_map = {"BTC": "BTC/BUSD"}
    assert ex._market_symbol("BTC") == "BTC/BUSD"


# ---------------------------------------------------------------------------
# get_position / get_cash
# ---------------------------------------------------------------------------

def test_get_position():
    ex = _make_executor()
    assert ex.get_position("BTC") == 1.5


def test_get_position_with_slash():
    ex = _make_executor()
    assert ex.get_position("BTC/USDT") == 1.5


def test_get_cash():
    ex = _make_executor()
    assert ex.get_cash() == 10000.0


def test_get_position_exchange_error():
    mock_ex = _mock_exchange()
    mock_ex.fetch_balance.side_effect = Exception("network error")
    ex = _make_executor(mock_ex)
    assert ex.get_position("BTC") == 0.0


def test_get_cash_exchange_error():
    mock_ex = _mock_exchange()
    mock_ex.fetch_balance.side_effect = Exception("network error")
    ex = _make_executor(mock_ex)
    assert ex.get_cash() == 0.0


# ---------------------------------------------------------------------------
# submit_order — buy
# ---------------------------------------------------------------------------

def test_buy_order():
    ex = _make_executor()
    order = Order(symbol="BTC", side="buy", qty=0.1)
    fill = ex.submit_order(order)
    assert fill is not None
    assert fill.symbol == "BTC"
    assert fill.side == "buy"
    assert fill.qty == 0.1
    assert fill.price == 50002.0
    assert fill.order_id == "buy-001"
    ex._exchange.create_market_buy_order.assert_called_once()


def test_buy_order_no_asks():
    mock_ex = _mock_exchange()
    mock_ex.fetch_order_book.return_value = {"asks": [], "bids": [[49990, 1]]}
    ex = _make_executor(mock_ex)
    fill = ex.submit_order(Order(symbol="BTC", side="buy", qty=0.1))
    assert fill is None


def test_buy_order_insufficient_cash():
    """Buy should be skipped when available cash is less than order value."""
    mock_ex = _mock_exchange()
    # USDT free = $100, but order for 0.1 BTC @ $50000 = $5000
    mock_ex.fetch_balance.return_value = {
        "BTC": {"total": 0.0, "free": 0.0},
        "USDT": {"total": 100.0, "free": 100.0},
    }
    ex = _make_executor(mock_ex)
    fill = ex.submit_order(Order(symbol="BTC", side="buy", qty=0.1))
    assert fill is None
    mock_ex.create_market_buy_order.assert_not_called()


def test_buy_order_exceeds_book_depth():
    mock_ex = _mock_exchange()
    # Book depth = 50000*2 + 50010*3 + 50020*5 = 500,130
    # max_book_fraction = 0.1 → max order = 50,013
    # qty=1.5 → order_value = 1.5 * 50000 = 75,000 > 50,013
    ex = _make_executor(mock_ex)
    fill = ex.submit_order(Order(symbol="BTC", side="buy", qty=1.5))
    assert fill is None
    mock_ex.create_market_buy_order.assert_not_called()


# ---------------------------------------------------------------------------
# submit_order — sell
# ---------------------------------------------------------------------------

def test_sell_order():
    ex = _make_executor()
    order = Order(symbol="BTC", side="sell", qty=0.5)
    fill = ex.submit_order(order)
    assert fill is not None
    assert fill.symbol == "BTC"
    assert fill.side == "sell"
    assert fill.qty == 0.5
    assert fill.price == 49988.0
    assert fill.order_id == "sell-001"
    ex._exchange.create_market_sell_order.assert_called_once()


def test_sell_order_no_bids():
    mock_ex = _mock_exchange()
    mock_ex.fetch_order_book.return_value = {"asks": [[50000, 1]], "bids": []}
    ex = _make_executor(mock_ex)
    fill = ex.submit_order(Order(symbol="BTC", side="sell", qty=0.1))
    assert fill is None


def test_sell_order_exceeds_book_depth():
    mock_ex = _mock_exchange()
    ex = _make_executor(mock_ex)
    fill = ex.submit_order(Order(symbol="BTC", side="sell", qty=1.5))
    assert fill is None
    mock_ex.create_market_sell_order.assert_not_called()


# ---------------------------------------------------------------------------
# submit_order — error handling
# ---------------------------------------------------------------------------

def test_submit_order_unknown_side():
    ex = _make_executor()
    fill = ex.submit_order(Order(symbol="BTC", side="short", qty=0.1))
    assert fill is None


def test_submit_order_exchange_exception():
    mock_ex = _mock_exchange()
    mock_ex.fetch_order_book.side_effect = Exception("timeout")
    ex = _make_executor(mock_ex)
    fill = ex.submit_order(Order(symbol="BTC", side="buy", qty=0.1))
    assert fill is None


# ---------------------------------------------------------------------------
# rebalance (inherited from Executor ABC)
# ---------------------------------------------------------------------------

def test_rebalance():
    mock_ex = _mock_exchange()
    # Ensure enough cash for buy: 0.5 BTC @ ~$50k = $25k
    mock_ex.fetch_balance.return_value = {
        "BTC": {"total": 1.5, "free": 1.5},
        "USDT": {"total": 50000.0, "free": 50000.0},
    }
    ex = _make_executor(mock_ex)
    # Current BTC position is 1.5 (from mock).
    # Target 2.0 → buy 0.5.
    fills = ex.rebalance({"BTC": 2.0})
    assert len(fills) == 1
    assert fills[0].side == "buy"


def test_rebalance_sell():
    ex = _make_executor()
    # Current 1.5 → target 1.0 → sell 0.5.
    fills = ex.rebalance({"BTC": 1.0})
    assert len(fills) == 1
    assert fills[0].side == "sell"


def test_rebalance_no_change():
    ex = _make_executor()
    fills = ex.rebalance({"BTC": 1.5})
    assert len(fills) == 0
