"""Tests for PolymarketClient — all HTTP calls are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from alpha_os.data.polymarket_client import (
    PolymarketClient,
    PolymarketMarket,
    PolymarketOrderbook,
)


@pytest.fixture
def client():
    return PolymarketClient(
        gamma_url="https://gamma-api.example.com",
        clob_url="https://clob.example.com",
        timeout=5,
    )


def _gamma_market_response():
    return [
        {
            "conditionId": "cond-001",
            "question": "Will event X happen?",
            "tokens": [
                {"token_id": "token-yes-001", "price": 0.65},
                {"token_id": "token-no-001", "price": 0.35},
            ],
            "volumeNum": 50000.0,
            "liquidityNum": 25000.0,
            "endDate": "2026-06-01T00:00:00Z",
            "active": True,
            "slug": "will-event-x-happen",
        },
        {
            "conditionId": "cond-002",
            "question": "Will event Y happen?",
            "tokens": [
                {"token_id": "token-yes-002", "price": 0.80},
                {"token_id": "token-no-002", "price": 0.20},
            ],
            "volumeNum": 5000.0,
            "liquidityNum": 500.0,
            "endDate": "2026-07-01T00:00:00Z",
            "active": True,
            "slug": "will-event-y-happen",
        },
    ]


def _clob_orderbook_response():
    return {
        "bids": [
            {"price": 0.64, "size": 100.0},
            {"price": 0.63, "size": 200.0},
        ],
        "asks": [
            {"price": 0.66, "size": 150.0},
            {"price": 0.67, "size": 250.0},
        ],
    }


class TestListMarkets:
    def test_returns_parsed_markets(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _gamma_market_response()
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._session, "get", return_value=mock_resp):
            markets = client.list_markets()

        assert len(markets) == 2
        assert markets[0].condition_id == "cond-001"
        assert markets[0].question == "Will event X happen?"
        assert markets[0].token_ids == ["token-yes-001", "token-no-001"]
        assert markets[0].outcome_prices == [0.65, 0.35]
        assert markets[0].volume_usd == 50000.0
        assert markets[0].liquidity_usd == 25000.0

    def test_filters_by_min_liquidity(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _gamma_market_response()
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._session, "get", return_value=mock_resp):
            markets = client.list_markets(min_liquidity_usd=10000.0)

        assert len(markets) == 1
        assert markets[0].condition_id == "cond-001"

    def test_handles_api_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            markets = client.list_markets()

        assert markets == []


class TestGetMarket:
    def test_returns_single_market(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _gamma_market_response()[0]
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._session, "get", return_value=mock_resp):
            market = client.get_market("cond-001")

        assert market is not None
        assert market.condition_id == "cond-001"
        assert len(market.token_ids) == 2

    def test_returns_none_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("not found")):
            market = client.get_market("bad-id")

        assert market is None


class TestGetOrderbook:
    def test_returns_parsed_orderbook(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _clob_orderbook_response()
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._session, "get", return_value=mock_resp):
            ob = client.get_orderbook("token-yes-001")

        assert ob is not None
        assert ob.token_id == "token-yes-001"
        assert len(ob.bids) == 2
        assert len(ob.asks) == 2
        assert ob.bids[0][0] == 0.64  # best bid
        assert ob.asks[0][0] == 0.66  # best ask
        assert ob.mid_price == pytest.approx(0.65)

    def test_returns_none_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            ob = client.get_orderbook("token-yes-001")

        assert ob is None


class TestGetPrice:
    def test_returns_mid_price(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _clob_orderbook_response()
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._session, "get", return_value=mock_resp):
            price = client.get_price("token-yes-001")

        assert price == pytest.approx(0.65)

    def test_get_prices_multiple(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _clob_orderbook_response()
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client._session, "get", return_value=mock_resp):
            prices = client.get_prices(["token-yes-001", "token-yes-002"])

        assert len(prices) == 2
        assert all(isinstance(v, float) for v in prices.values())
