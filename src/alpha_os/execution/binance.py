"""Binance spot executor via CCXT with orderbook depth and slippage protection."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from .executor import Executor, Order, Fill

logger = logging.getLogger(__name__)

_SECRETS_FILE = "binance"


def _load_secrets(name: str = _SECRETS_FILE) -> dict[str, str]:
    """Load key-value pairs from ~/.secrets/{name}."""
    secret_path = Path.home() / ".secrets" / name
    if not secret_path.exists():
        return {}
    result: dict[str, str] = {}
    for line in secret_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            key, val = line.split("=", 1)
            result[key.strip()] = val.strip().strip("'\"")
    return result


def _get_credentials() -> tuple[str, str]:
    """Return (api_key, secret) from env vars or ~/.secrets/binance."""
    api_key = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_SECRET_KEY", "")
    if api_key and secret:
        return api_key, secret
    secrets = _load_secrets()
    api_key = api_key or secrets.get("BINANCE_API_KEY", "")
    secret = secret or secrets.get("BINANCE_SECRET_KEY", "")
    return api_key, secret


def create_spot_exchange(testnet: bool = True):
    """Create authenticated Binance spot exchange via CCXT."""
    import ccxt

    api_key, secret = _get_credentials()
    if not api_key or not secret:
        raise ValueError(
            "Binance API credentials not found. "
            "Set BINANCE_API_KEY/BINANCE_SECRET_KEY env vars "
            "or create ~/.secrets/binance"
        )
    exchange = ccxt.binance(
        {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
    )
    if testnet:
        exchange.set_sandbox_mode(True)
    logger.info("Created Binance spot exchange (testnet=%s)", testnet)
    return exchange


class BinanceExecutor(Executor):
    """Live Binance spot executor with orderbook depth check and slippage guard.

    Conforms to alpha-os Executor interface. Positions and cash are always
    queried from the exchange (single source of truth).
    """

    def __init__(
        self,
        testnet: bool = True,
        symbol_map: dict[str, str] | None = None,
        max_slippage_bps: float = 10.0,
        max_book_fraction: float = 0.1,
    ) -> None:
        self._exchange = create_spot_exchange(testnet=testnet)
        self._testnet = testnet
        # Map alpha-os symbol names to CCXT market symbols.
        # e.g. {"BTC": "BTC/USDT"} — default appends /USDT.
        self._symbol_map = symbol_map or {}
        self._max_slippage_bps = max_slippage_bps
        self._max_book_fraction = max_book_fraction

    def _market_symbol(self, symbol: str) -> str:
        if symbol in self._symbol_map:
            return self._symbol_map[symbol]
        if "/" not in symbol:
            return f"{symbol}/USDT"
        return symbol

    def submit_order(self, order: Order) -> Fill | None:
        market = self._market_symbol(order.symbol)
        try:
            if order.side == "buy":
                return self._market_buy(market, order)
            elif order.side == "sell":
                return self._market_sell(market, order)
            else:
                logger.error("Unknown side: %s", order.side)
                return None
        except Exception as e:
            logger.error("Order failed for %s: %s", market, e)
            return None

    def get_position(self, symbol: str) -> float:
        base = symbol.split("/")[0] if "/" in symbol else symbol
        try:
            balance = self._exchange.fetch_balance()
            return float(balance.get(base, {}).get("total", 0.0))
        except Exception as e:
            logger.error("Failed to fetch position for %s: %s", base, e)
            return 0.0

    def get_cash(self) -> float:
        try:
            balance = self._exchange.fetch_balance()
            return float(balance.get("USDT", {}).get("free", 0.0))
        except Exception as e:
            logger.error("Failed to fetch USDT balance: %s", e)
            return 0.0

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value from exchange (USDT + marked positions)."""
        try:
            balance = self._exchange.fetch_balance()
            total = float(balance.get("USDT", {}).get("total", 0.0))
            for asset, amounts in balance.items():
                if asset in ("USDT", "free", "used", "total", "info"):
                    continue
                if not isinstance(amounts, dict):
                    continue
                qty = float(amounts.get("total", 0.0))
                if qty > 1e-8:
                    try:
                        market = f"{asset}/USDT"
                        ticker = self._exchange.fetch_ticker(market)
                        total += qty * float(ticker["last"])
                    except Exception:
                        pass
            return total
        except Exception as e:
            logger.error("Failed to fetch portfolio value: %s", e)
            return 0.0

    @property
    def all_positions(self) -> dict[str, float]:
        """All non-zero positions from exchange."""
        try:
            balance = self._exchange.fetch_balance()
            positions: dict[str, float] = {}
            for asset, amounts in balance.items():
                if asset in ("USDT", "free", "used", "total", "info"):
                    continue
                if not isinstance(amounts, dict):
                    continue
                qty = float(amounts.get("total", 0.0))
                if qty > 1e-8:
                    positions[asset] = qty
            return positions
        except Exception as e:
            logger.error("Failed to fetch positions: %s", e)
            return {}

    def fetch_ticker_price(self, symbol: str) -> float | None:
        """Fetch real-time last price from Binance via CCXT."""
        market = self._market_symbol(symbol)
        try:
            ticker = self._exchange.fetch_ticker(market)
            return float(ticker["last"])
        except Exception as e:
            logger.warning("Failed to fetch ticker for %s: %s", market, e)
            return None

    # ------------------------------------------------------------------

    def _market_buy(self, market: str, order: Order) -> Fill | None:
        orderbook = self._exchange.fetch_order_book(market, limit=5)
        best_ask = orderbook["asks"][0][0] if orderbook["asks"] else None
        if best_ask is None:
            logger.error("No asks in orderbook for %s", market)
            return None

        # Depth check: reject if order value exceeds fraction of visible book.
        order_value = order.qty * best_ask
        book_depth = sum(p * q for p, q in orderbook["asks"][:5])
        if book_depth > 0 and order_value > book_depth * self._max_book_fraction:
            logger.warning(
                "Buy $%.2f > %.0f%% of book depth $%.2f for %s, skipping",
                order_value,
                self._max_book_fraction * 100,
                book_depth,
                market,
            )
            return None

        qty = float(self._exchange.amount_to_precision(market, order.qty))
        if qty <= 0:
            return None

        t0 = time.perf_counter()
        result = self._exchange.create_market_buy_order(market, qty)
        latency_ms = (time.perf_counter() - t0) * 1000

        filled_price = float(result.get("average", best_ask))
        filled_qty = float(result.get("filled", qty))

        slippage = abs(filled_price - best_ask) / best_ask * 10_000
        if slippage > self._max_slippage_bps:
            logger.warning("High slippage: %.1f bps on %s buy", slippage, market)

        return Fill(
            symbol=order.symbol,
            side="buy",
            qty=filled_qty,
            price=filled_price,
            order_id=str(result.get("id", "")),
            slippage_bps=slippage,
            latency_ms=latency_ms,
        )

    def _market_sell(self, market: str, order: Order) -> Fill | None:
        orderbook = self._exchange.fetch_order_book(market, limit=5)
        best_bid = orderbook["bids"][0][0] if orderbook["bids"] else None
        if best_bid is None:
            logger.error("No bids in orderbook for %s", market)
            return None

        order_value = order.qty * best_bid
        book_depth = sum(p * q for p, q in orderbook["bids"][:5])
        if book_depth > 0 and order_value > book_depth * self._max_book_fraction:
            logger.warning(
                "Sell $%.2f > %.0f%% of book depth $%.2f for %s, skipping",
                order_value,
                self._max_book_fraction * 100,
                book_depth,
                market,
            )
            return None

        qty = float(self._exchange.amount_to_precision(market, order.qty))
        if qty <= 0:
            return None

        t0 = time.perf_counter()
        result = self._exchange.create_market_sell_order(market, qty)
        latency_ms = (time.perf_counter() - t0) * 1000

        filled_price = float(result.get("average", best_bid))
        filled_qty = float(result.get("filled", qty))

        slippage = abs(filled_price - best_bid) / best_bid * 10_000
        if slippage > self._max_slippage_bps:
            logger.warning("High slippage: %.1f bps on %s sell", slippage, market)

        return Fill(
            symbol=order.symbol,
            side="sell",
            qty=filled_qty,
            price=filled_price,
            order_id=str(result.get("id", "")),
            slippage_bps=slippage,
            latency_ms=latency_ms,
        )
