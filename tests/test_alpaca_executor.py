"""Tests for AlpacaExecutor — all HTTP calls are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from alpha_os.execution.alpaca import (
    AlpacaExecutor,
    _get_credentials,
    _load_secrets,
)
from alpha_os.execution.costs import ExecutionCostModel
from alpha_os.execution.executor import Order


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def test_load_secrets_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.execution.alpaca.Path.home", lambda: tmp_path)
    assert _load_secrets() == {}


def test_load_secrets_parses_file(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os.execution.alpaca.Path.home", lambda: tmp_path)
    secrets_dir = tmp_path / ".secrets"
    secrets_dir.mkdir()
    (secrets_dir / "alpaca").write_text(
        "export ALPACA_API_KEY='key123'\nALPACA_SECRET_KEY=secret456\n"
    )
    result = _load_secrets()
    assert result == {"ALPACA_API_KEY": "key123", "ALPACA_SECRET_KEY": "secret456"}


def test_get_credentials_from_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "envkey")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "envsecret")
    key, secret = _get_credentials()
    assert key == "envkey"
    assert secret == "envsecret"


# ---------------------------------------------------------------------------
# Helpers to build a mock executor
# ---------------------------------------------------------------------------

def _make_executor(session=None, initial_capital=10000.0):
    """Create AlpacaExecutor bypassing real session init."""
    executor = object.__new__(AlpacaExecutor)
    executor._session = session or MagicMock()
    executor._base_url = "https://paper-api.alpaca.markets"
    executor._paper = True
    executor._cost_model = ExecutionCostModel(commission_pct=0.0, modeled_slippage_pct=0.05)
    executor._max_slippage_bps = 20.0
    executor._managed_cash = initial_capital
    executor._managed_positions = {}
    executor._initial_capital = initial_capital
    executor._fills = []
    executor._reconciliation_cash_offset = 0.0
    executor._reconciliation_position_offsets = {}
    return executor


def _mock_session():
    session = MagicMock()
    post_resp = MagicMock()
    post_resp.json.return_value = {
        "id": "order-001",
        "filled_avg_price": "150.50",
        "filled_qty": "10",
        "status": "filled",
    }
    post_resp.raise_for_status = MagicMock()
    session.post.return_value = post_resp

    get_resp = MagicMock()
    get_resp.json.return_value = {"cash": "50000.00", "qty": "100"}
    get_resp.raise_for_status = MagicMock()
    get_resp.status_code = 200
    session.get.return_value = get_resp
    return session


# ---------------------------------------------------------------------------
# Order tests
# ---------------------------------------------------------------------------

class TestSubmitOrder:
    def test_buy_order(self):
        session = _mock_session()
        executor = _make_executor(session=session)

        order = Order(symbol="NVDA", side="buy", qty=10.0)
        fill = executor.submit_order(order)

        assert fill is not None
        assert fill.symbol == "NVDA"
        assert fill.side == "buy"
        assert fill.price == 150.50
        assert fill.order_id == "order-001"

    def test_buy_updates_managed_state(self):
        session = _mock_session()
        executor = _make_executor(session=session, initial_capital=10000.0)

        order = Order(symbol="NVDA", side="buy", qty=10.0)
        executor.submit_order(order)

        assert executor.get_position("NVDA") == 10.0
        assert executor.get_cash() < 10000.0

    def test_sell_without_position_rejected(self):
        executor = _make_executor()

        order = Order(symbol="NVDA", side="sell", qty=5.0)
        fill = executor.submit_order(order)

        assert fill is None

    def test_api_failure_returns_none(self):
        session = MagicMock()
        session.post.side_effect = Exception("API down")
        executor = _make_executor(session=session)

        order = Order(symbol="NVDA", side="buy", qty=10.0)
        fill = executor.submit_order(order)

        assert fill is None


class TestPortfolio:
    def test_all_positions_filters_zero(self):
        executor = _make_executor()
        executor._managed_positions = {"NVDA": 5.0, "AAPL": 0.0, "MSFT": 1e-10}
        assert executor.all_positions == {"NVDA": 5.0}

    def test_supports_short_false(self):
        executor = _make_executor()
        assert executor.supports_short is False
