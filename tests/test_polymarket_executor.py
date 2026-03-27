"""Tests for PolymarketExecutor — all CLOB calls are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from alpha_os_recovery.execution.costs import PolymarketCostModel
from alpha_os_recovery.execution.executor import Order
from alpha_os_recovery.execution.polymarket import (
    PolymarketExecutor,
    _get_credentials,
    _load_secrets,
)


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def test_load_secrets_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os_recovery.execution.secrets.Path.home", lambda: tmp_path)
    assert _load_secrets("polymarket") == {}


def test_load_secrets_parses_file(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha_os_recovery.execution.secrets.Path.home", lambda: tmp_path)
    secrets_dir = tmp_path / ".secrets"
    secrets_dir.mkdir()
    (secrets_dir / "polymarket").write_text(
        "export POLYMARKET_PRIVATE_KEY='0xabc123'\nPOLYMARKET_API_KEY=apikey456\n"
    )
    result = _load_secrets("polymarket")
    assert result == {
        "POLYMARKET_PRIVATE_KEY": "0xabc123",
        "POLYMARKET_API_KEY": "apikey456",
    }


def test_get_credentials_from_env(monkeypatch):
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xenvkey")
    monkeypatch.setenv("POLYMARKET_API_KEY", "envapikey")
    key, api_key = _get_credentials()
    assert key == "0xenvkey"
    assert api_key == "envapikey"


# ---------------------------------------------------------------------------
# Helpers to build a mock executor
# ---------------------------------------------------------------------------

def _make_executor(client=None, initial_capital=1000.0):
    """Create PolymarketExecutor bypassing real CLOB client init."""
    executor = object.__new__(PolymarketExecutor)
    executor._client = client or MagicMock()
    executor._host = "https://clob.polymarket.com"
    executor._max_position_per_market = 100.0
    executor._cost_model = PolymarketCostModel()
    executor._managed_cash = initial_capital
    executor._managed_positions = {}
    executor._initial_capital = initial_capital
    executor._fills = []
    return executor


def _mock_clob_client():
    client = MagicMock()
    client.get_last_trade_price.return_value = {"price": 0.65}
    client.create_and_post_order.return_value = {
        "orderID": "order-001",
        "status": "matched",
    }
    client.get_market.return_value = {
        "resolved": True,
        "outcome_price": 1.0,
    }
    return client


# ---------------------------------------------------------------------------
# Basic order tests
# ---------------------------------------------------------------------------

class TestSubmitOrder:
    def test_buy_order(self):
        client = _mock_clob_client()
        executor = _make_executor(client=client)

        order = Order(symbol="token-yes-001", side="buy", qty=10.0)
        fill = executor.submit_order(order)

        assert fill is not None
        assert fill.symbol == "token-yes-001"
        assert fill.side == "buy"
        assert fill.qty == 10.0
        assert fill.price == 0.65
        assert fill.order_id == "order-001"

    def test_buy_updates_managed_state(self):
        client = _mock_clob_client()
        executor = _make_executor(client=client, initial_capital=1000.0)

        order = Order(symbol="token-yes-001", side="buy", qty=10.0)
        executor.submit_order(order)

        assert executor.get_position("token-yes-001") == 10.0
        assert executor.get_cash() < 1000.0

    def test_sell_without_position_rejected(self):
        executor = _make_executor()

        order = Order(symbol="token-yes-001", side="sell", qty=5.0)
        fill = executor.submit_order(order)

        assert fill is None

    def test_exceeds_max_position_rejected(self):
        client = _mock_clob_client()
        client.get_last_trade_price.return_value = {"price": 0.90}
        executor = _make_executor(client=client)
        executor._max_position_per_market = 50.0

        order = Order(symbol="token-yes-001", side="buy", qty=100.0)
        fill = executor.submit_order(order)

        assert fill is None

    def test_price_failure_returns_none(self):
        client = _mock_clob_client()
        client.get_last_trade_price.side_effect = Exception("API down")
        executor = _make_executor(client=client)

        order = Order(symbol="token-yes-001", side="buy", qty=10.0)
        fill = executor.submit_order(order)

        assert fill is None


# ---------------------------------------------------------------------------
# Portfolio and settlement
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_portfolio_value(self):
        client = _mock_clob_client()
        executor = _make_executor(client=client, initial_capital=1000.0)
        executor._managed_positions = {"token-001": 10.0}

        value = executor.portfolio_value
        assert value == 1000.0 + 10.0 * 0.65

    def test_all_positions_filters_zero(self):
        executor = _make_executor()
        executor._managed_positions = {"a": 5.0, "b": 0.0, "c": 1e-10}
        assert executor.all_positions == {"a": 5.0}

    def test_settle_resolved_market(self):
        client = _mock_clob_client()
        executor = _make_executor(client=client, initial_capital=500.0)
        executor._managed_positions = {"token-001": 20.0}

        payout = executor.settle_market("token-001")

        assert payout == 20.0  # outcome_price=1.0
        assert executor.get_cash() == 520.0
        assert executor.get_position("token-001") == 0.0


class TestCostModel:
    def test_taker_cost(self):
        model = PolymarketCostModel(taker_fee_pct=1.5)
        cost = model.estimate_order_cost(100.0, is_maker=False)
        assert cost.commission == pytest.approx(1.5)

    def test_maker_cost_zero(self):
        model = PolymarketCostModel(maker_fee_pct=0.0)
        cost = model.estimate_order_cost(100.0, is_maker=True)
        assert cost.commission == pytest.approx(0.0)
