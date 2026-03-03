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


def _get_credentials(testnet: bool = True) -> tuple[str, str]:
    """Return (api_key, secret) from env vars or ~/.secrets/binance{_real}."""
    api_key = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_SECRET_KEY", "")
    if api_key and secret:
        return api_key, secret
    secrets_name = _SECRETS_FILE if testnet else f"{_SECRETS_FILE}_real"
    secrets = _load_secrets(secrets_name)
    api_key = api_key or secrets.get("BINANCE_API_KEY", "")
    secret = secret or secrets.get("BINANCE_SECRET_KEY", "")
    return api_key, secret


def create_spot_exchange(testnet: bool = True):
    """Create authenticated Binance spot exchange via CCXT."""
    import ccxt

    api_key, secret = _get_credentials(testnet=testnet)
    secrets_file = "binance" if testnet else "binance_real"
    if not api_key or not secret:
        raise ValueError(
            "Binance API credentials not found. "
            "Set BINANCE_API_KEY/BINANCE_SECRET_KEY env vars "
            f"or create ~/.secrets/{secrets_file}"
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
    """Live Binance spot executor with managed position tracking.

    Tracks only positions created by alpha-os trades, separate from the
    full exchange account balance. Exchange is used for order execution
    and pre-trade safety checks, but portfolio_value reflects managed
    equity only.
    """

    def __init__(
        self,
        testnet: bool = True,
        symbol_map: dict[str, str] | None = None,
        max_slippage_bps: float = 10.0,
        max_book_fraction: float = 0.1,
        optimizer: object | None = None,
        initial_capital: float = 10000.0,
    ) -> None:
        self._exchange = create_spot_exchange(testnet=testnet)
        self._testnet = testnet
        self._symbol_map = symbol_map or {}
        self._max_slippage_bps = max_slippage_bps
        self._max_book_fraction = max_book_fraction
        self._optimizer = optimizer
        # Managed state: tracks only alpha-os positions, not full exchange
        self._managed_cash: float = initial_capital
        self._managed_positions: dict[str, float] = {}
        self._initial_capital = initial_capital

    def _market_symbol(self, symbol: str) -> str:
        if symbol in self._symbol_map:
            return self._symbol_map[symbol]
        if "/" not in symbol:
            return f"{symbol}/USDT"
        return symbol

    def _update_managed_state(self, fill: Fill) -> None:
        """Update internal managed position tracking after a successful fill."""
        cost = fill.qty * fill.price
        if fill.side == "buy":
            self._managed_cash -= cost
            self._managed_positions[fill.symbol] = (
                self._managed_positions.get(fill.symbol, 0.0) + fill.qty
            )
        elif fill.side == "sell":
            self._managed_cash += cost
            self._managed_positions[fill.symbol] = (
                self._managed_positions.get(fill.symbol, 0.0) - fill.qty
            )

    def submit_order(self, order: Order) -> Fill | None:
        if self._optimizer is not None:
            for attempt in range(3):
                if self._optimizer.optimal_execution_window(order.side):
                    break
                logger.info(
                    "Order deferred by optimizer (attempt %d/3): %s %s %.6f",
                    attempt + 1, order.side, order.symbol, order.qty,
                )
                time.sleep(30)
            else:
                logger.warning(
                    "Optimizer blocked after 3 attempts, executing anyway: %s %s",
                    order.side, order.symbol,
                )

            slices = self._optimizer.split_order(order.qty)
            if len(slices) > 1:
                fill = self._execute_slices(order, slices)
                if fill is not None:
                    self._update_managed_state(fill)
                return fill

        fill = self._submit_single(order)
        if fill is not None:
            self._update_managed_state(fill)
        return fill

    def _submit_single(self, order: Order) -> Fill | None:
        """Submit a single order without optimizer checks."""
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

    def _execute_slices(self, order: Order, slices: list[float]) -> Fill | None:
        """Execute order in multiple slices, aggregate into a single Fill."""
        logger.info(
            "Splitting %s %s into %d slices", order.side, order.symbol, len(slices),
        )
        total_qty = 0.0
        total_cost = 0.0
        total_latency = 0.0
        total_slippage = 0.0

        for i, slice_qty in enumerate(slices):
            slice_order = Order(symbol=order.symbol, side=order.side, qty=slice_qty)
            fill = self._submit_single(slice_order)
            if fill is not None:
                total_qty += fill.qty
                total_cost += fill.qty * fill.price
                total_latency += fill.latency_ms
                total_slippage += fill.slippage_bps * fill.qty

        if total_qty == 0:
            return None

        avg_price = total_cost / total_qty
        avg_slippage = total_slippage / total_qty
        return Fill(
            symbol=order.symbol,
            side=order.side,
            qty=total_qty,
            price=avg_price,
            slippage_bps=avg_slippage,
            latency_ms=total_latency,
        )

    # ── Managed state interface (used by trader for sizing/risk/PnL) ──

    def get_position(self, symbol: str) -> float:
        return self._managed_positions.get(symbol, 0.0)

    def get_cash(self) -> float:
        return self._managed_cash

    @property
    def portfolio_value(self) -> float:
        """Portfolio value from managed positions only (not whole exchange)."""
        total = self._managed_cash
        for symbol, qty in self._managed_positions.items():
            if abs(qty) > 1e-8:
                price = self.fetch_ticker_price(symbol)
                if price is not None:
                    total += qty * price
        return total

    @property
    def all_positions(self) -> dict[str, float]:
        return {s: q for s, q in self._managed_positions.items() if abs(q) > 1e-8}

    # ── Exchange-level access (for safety checks and reconciliation) ──

    def _exchange_cash(self) -> float:
        """Fetch USDT balance from exchange (for pre-trade safety check)."""
        try:
            balance = self._exchange.fetch_balance()
            return float(balance.get("USDT", {}).get("free", 0.0))
        except Exception as e:
            logger.error("Failed to fetch USDT balance: %s", e)
            return 0.0

    def _exchange_position(self, symbol: str) -> float:
        """Fetch actual position from exchange (for reconciliation)."""
        base = symbol.split("/")[0] if "/" in symbol else symbol
        try:
            balance = self._exchange.fetch_balance()
            return float(balance.get(base, {}).get("total", 0.0))
        except Exception as e:
            logger.error("Failed to fetch position for %s: %s", base, e)
            return 0.0

    def fetch_ticker_price(self, symbol: str) -> float | None:
        """Fetch real-time last price from Binance via CCXT."""
        market = self._market_symbol(symbol)
        try:
            ticker = self._exchange.fetch_ticker(market)
            return float(ticker["last"])
        except Exception as e:
            logger.warning("Failed to fetch ticker for %s: %s", market, e)
            return None

    # ── Order execution (exchange interaction) ──

    def _market_buy(self, market: str, order: Order) -> Fill | None:
        orderbook = self._exchange.fetch_order_book(market, limit=5)
        best_ask = orderbook["asks"][0][0] if orderbook["asks"] else None
        if best_ask is None:
            logger.error("No asks in orderbook for %s", market)
            return None

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

        # Check actual exchange balance (not managed cash)
        available_cash = self._exchange_cash()
        if order_value > available_cash:
            logger.warning(
                "Insufficient cash: need $%.2f, have $%.2f for %s, skipping",
                order_value, available_cash, market,
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
