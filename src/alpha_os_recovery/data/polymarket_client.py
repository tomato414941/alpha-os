"""Polymarket data client — market discovery and price data via Gamma + CLOB APIs."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"


@dataclass(frozen=True)
class PolymarketMarket:
    """Snapshot of a Polymarket prediction market."""
    condition_id: str
    question: str
    token_ids: list[str]
    outcome_prices: list[float]
    volume_usd: float
    liquidity_usd: float
    end_date: str
    active: bool
    slug: str = ""


@dataclass(frozen=True)
class PolymarketOrderbook:
    """Simplified orderbook for a Polymarket token."""
    token_id: str
    bids: list[tuple[float, float]]  # (price, size)
    asks: list[tuple[float, float]]
    mid_price: float


class PolymarketClient:
    """Client for Polymarket Gamma API (discovery) and CLOB API (orderbook).

    Uses unauthenticated endpoints only — no API keys required for reads.
    """

    def __init__(
        self,
        gamma_url: str = GAMMA_API_URL,
        clob_url: str = CLOB_API_URL,
        timeout: int = 15,
    ) -> None:
        self._gamma_url = gamma_url.rstrip("/")
        self._clob_url = clob_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    def list_markets(
        self,
        *,
        active: bool = True,
        min_liquidity_usd: float = 0.0,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PolymarketMarket]:
        """Discover markets via Gamma API."""
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "order": "liquidity",
            "ascending": "false",
        }
        try:
            resp = self._session.get(
                f"{self._gamma_url}/markets",
                params=params,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            raw_markets = resp.json()
        except Exception as e:
            logger.error("Failed to list Polymarket markets: %s", e)
            return []

        markets: list[PolymarketMarket] = []
        for m in raw_markets:
            liquidity = float(m.get("liquidityNum", m.get("liquidity", 0)) or 0)
            if liquidity < min_liquidity_usd:
                continue

            token_ids = []
            outcome_prices = []
            tokens = m.get("tokens", [])
            if isinstance(tokens, list):
                for t in tokens:
                    if isinstance(t, dict):
                        token_ids.append(str(t.get("token_id", "")))
                        outcome_prices.append(float(t.get("price", 0) or 0))

            markets.append(PolymarketMarket(
                condition_id=str(m.get("conditionId", m.get("condition_id", ""))),
                question=str(m.get("question", "")),
                token_ids=token_ids,
                outcome_prices=outcome_prices,
                volume_usd=float(m.get("volumeNum", m.get("volume", 0)) or 0),
                liquidity_usd=liquidity,
                end_date=str(m.get("endDate", m.get("end_date_iso", ""))),
                active=bool(m.get("active", True)),
                slug=str(m.get("slug", "")),
            ))

        return markets

    def get_market(self, condition_id: str) -> PolymarketMarket | None:
        """Get a single market by condition ID."""
        try:
            resp = self._session.get(
                f"{self._gamma_url}/markets/{condition_id}",
                timeout=self._timeout,
            )
            resp.raise_for_status()
            m = resp.json()
        except Exception as e:
            logger.error("Failed to get market %s: %s", condition_id, e)
            return None

        token_ids = []
        outcome_prices = []
        tokens = m.get("tokens", [])
        if isinstance(tokens, list):
            for t in tokens:
                if isinstance(t, dict):
                    token_ids.append(str(t.get("token_id", "")))
                    outcome_prices.append(float(t.get("price", 0) or 0))

        return PolymarketMarket(
            condition_id=str(m.get("conditionId", m.get("condition_id", ""))),
            question=str(m.get("question", "")),
            token_ids=token_ids,
            outcome_prices=outcome_prices,
            volume_usd=float(m.get("volumeNum", m.get("volume", 0)) or 0),
            liquidity_usd=float(m.get("liquidityNum", m.get("liquidity", 0)) or 0),
            end_date=str(m.get("endDate", m.get("end_date_iso", ""))),
            active=bool(m.get("active", True)),
            slug=str(m.get("slug", "")),
        )

    def get_orderbook(self, token_id: str) -> PolymarketOrderbook | None:
        """Get CLOB orderbook for a specific token."""
        try:
            resp = self._session.get(
                f"{self._clob_url}/book",
                params={"token_id": token_id},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("Failed to get orderbook for %s: %s", token_id, e)
            return None

        bids: list[tuple[float, float]] = []
        asks: list[tuple[float, float]] = []

        for entry in data.get("bids", []):
            if isinstance(entry, dict):
                bids.append((float(entry.get("price", 0)), float(entry.get("size", 0))))
        for entry in data.get("asks", []):
            if isinstance(entry, dict):
                asks.append((float(entry.get("price", 0)), float(entry.get("size", 0))))

        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])

        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 1.0
        mid_price = (best_bid + best_ask) / 2.0

        return PolymarketOrderbook(
            token_id=token_id,
            bids=bids,
            asks=asks,
            mid_price=mid_price,
        )

    def get_price(self, token_id: str) -> float | None:
        """Get current mid price for a token from CLOB orderbook."""
        ob = self.get_orderbook(token_id)
        if ob is None:
            return None
        return ob.mid_price

    def get_prices(self, token_ids: list[str]) -> dict[str, float]:
        """Get prices for multiple tokens."""
        prices: dict[str, float] = {}
        for tid in token_ids:
            price = self.get_price(tid)
            if price is not None:
                prices[tid] = price
        return prices
