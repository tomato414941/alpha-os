"""Trade executor — abstract interface + Alpaca integration."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Order:
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    order_type: str = "market"


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    order_id: str = ""
    slippage_bps: float = 0.0
    latency_ms: float = 0.0


class Executor(ABC):
    """Abstract base for trade execution."""

    @abstractmethod
    def submit_order(self, order: Order) -> Fill | None:
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> float:
        ...

    @abstractmethod
    def get_cash(self) -> float:
        ...

    def set_price(self, symbol: str, price: float) -> None:
        """Set current price for a symbol. No-op for live executors."""
        pass

    def fetch_ticker_price(self, symbol: str) -> float | None:
        """Fetch real-time price from exchange. Returns None if unavailable."""
        return None

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value. Subclasses should override for accuracy."""
        return self.get_cash()

    @property
    def all_positions(self) -> dict[str, float]:
        """All current positions. Subclasses should override."""
        return {}

    @property
    def all_fills(self) -> list[Fill]:
        """All recorded fills. Subclasses should override."""
        return []

    def rebalance(self, target_shares: dict[str, float]) -> list[Fill]:
        """Rebalance to target positions, returning list of fills."""
        fills: list[Fill] = []
        for symbol, target in target_shares.items():
            current = self.get_position(symbol)
            delta = target - current
            if abs(delta) < 1e-6:
                continue
            side = "buy" if delta > 0 else "sell"
            order = Order(symbol=symbol, side=side, qty=abs(delta))
            fill = self.submit_order(order)
            if fill:
                fills.append(fill)
        return fills


class AlpacaExecutor(Executor):
    """Alpaca API executor (placeholder — requires API keys)."""

    def __init__(self, api_key: str, api_secret: str, base_url: str = ""):
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url or "https://paper-api.alpaca.markets"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import requests
                self._session = requests.Session()
                self._session.headers.update({
                    "APCA-API-KEY-ID": self._api_key,
                    "APCA-API-SECRET-KEY": self._api_secret,
                })
                self._client = True
            except ImportError:
                raise RuntimeError("requests library required for Alpaca executor")
        return self._session

    def submit_order(self, order: Order) -> Fill | None:
        session = self._get_client()
        try:
            resp = session.post(
                f"{self._base_url}/v2/orders",
                json={
                    "symbol": order.symbol,
                    "qty": str(order.qty),
                    "side": order.side,
                    "type": order.order_type,
                    "time_in_force": "day",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return Fill(
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                price=float(data.get("filled_avg_price", 0)),
                order_id=data.get("id", ""),
            )
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None

    def get_position(self, symbol: str) -> float:
        session = self._get_client()
        try:
            resp = session.get(f"{self._base_url}/v2/positions/{symbol}")
            if resp.status_code == 404:
                return 0.0
            resp.raise_for_status()
            return float(resp.json().get("qty", 0))
        except Exception:
            return 0.0

    def get_cash(self) -> float:
        session = self._get_client()
        try:
            resp = session.get(f"{self._base_url}/v2/account")
            resp.raise_for_status()
            return float(resp.json().get("cash", 0))
        except Exception:
            return 0.0
