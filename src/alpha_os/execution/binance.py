"""Binance spot executor via CCXT with orderbook depth and slippage protection."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from .constraints import ConstraintResult, apply_venue_constraints
from .costs import CostEstimate, ExecutionCostModel
from .executor import Executor, Order, Fill
from .planning import ExecutionIntent

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
        min_notional_buffer: float = 1.02,
        cost_model: ExecutionCostModel | None = None,
    ) -> None:
        self._exchange = create_spot_exchange(testnet=testnet)
        self._testnet = testnet
        self._symbol_map = symbol_map or {}
        self._max_slippage_bps = max_slippage_bps
        self._max_book_fraction = max_book_fraction
        self._optimizer = optimizer
        self._min_notional_buffer = min_notional_buffer
        self._cost_model = cost_model or ExecutionCostModel()
        self._min_notional_cache: dict[str, float | None] = {}
        # Managed state: tracks only alpha-os positions, not full exchange
        self._managed_cash: float = initial_capital
        self._managed_positions: dict[str, float] = {}
        self._initial_capital = initial_capital
        self._reconciliation_cash_offset: float = 0.0
        self._reconciliation_position_offsets: dict[str, float] = {}

    def _get_market_info(self, market: str) -> dict[str, Any] | None:
        market_fn = getattr(self._exchange, "market", None)
        if callable(market_fn):
            try:
                info = market_fn(market)
                if isinstance(info, dict):
                    return info
            except Exception:
                pass

        load_markets_fn = getattr(self._exchange, "load_markets", None)
        if callable(load_markets_fn):
            try:
                markets = load_markets_fn()
                if isinstance(markets, dict):
                    info = markets.get(market)
                    if isinstance(info, dict):
                        return info
            except Exception:
                pass

        markets_attr = getattr(self._exchange, "markets", None)
        if isinstance(markets_attr, dict):
            info = markets_attr.get(market)
            if isinstance(info, dict):
                return info
        return None

    @staticmethod
    def _extract_min_notional(market_info: dict[str, Any]) -> float | None:
        candidates: list[float] = []

        limits = market_info.get("limits", {})
        if isinstance(limits, dict):
            cost = limits.get("cost", {})
            if isinstance(cost, dict):
                cost_min = cost.get("min")
                if cost_min is not None:
                    try:
                        value = float(cost_min)
                        if value > 0:
                            candidates.append(value)
                    except (TypeError, ValueError):
                        pass

        info = market_info.get("info", {})
        if isinstance(info, dict):
            filters = info.get("filters", [])
            if isinstance(filters, list):
                for f in filters:
                    if not isinstance(f, dict):
                        continue
                    f_type = str(f.get("filterType", "")).upper()
                    if f_type not in {"MIN_NOTIONAL", "NOTIONAL"}:
                        continue
                    raw = f.get("minNotional") or f.get("notional")
                    if raw is None:
                        continue
                    try:
                        value = float(raw)
                        if value > 0:
                            candidates.append(value)
                    except (TypeError, ValueError):
                        pass

        if not candidates:
            return None
        return max(candidates)

    def _min_notional(self, market: str) -> float | None:
        if market in self._min_notional_cache:
            return self._min_notional_cache[market]

        market_info = self._get_market_info(market)
        if market_info is None:
            self._min_notional_cache[market] = None
            return None

        min_notional = self._extract_min_notional(market_info)
        self._min_notional_cache[market] = min_notional
        return min_notional

    def _precise_qty(self, market: str, qty: float) -> float:
        try:
            return float(self._exchange.amount_to_precision(market, qty))
        except Exception:
            return 0.0

    def _meets_notional(
        self,
        market: str,
        qty: float,
        ref_price: float,
        *,
        log_failure: bool = True,
    ) -> bool:
        if qty <= 0 or ref_price <= 0:
            return False
        min_notional = self._min_notional(market)
        if min_notional is None:
            return True

        required = min_notional * self._min_notional_buffer
        actual = qty * ref_price
        if actual >= required:
            return True

        if log_failure:
            logger.warning(
                "Order below min notional for %s: $%.4f < $%.4f (qty=%.8f, price=%.2f)",
                market,
                actual,
                required,
                qty,
                ref_price,
            )
        return False

    def _adjust_slices_for_notional(
        self,
        order: Order,
        slices: list[float],
        ref_price: float,
    ) -> list[float]:
        if len(slices) <= 1 or ref_price <= 0:
            return slices

        market = self._market_symbol(order.symbol)
        min_notional = self._min_notional(market)
        if min_notional is None:
            return slices

        total_notional = order.qty * ref_price
        required_slice_notional = min_notional * self._min_notional_buffer
        if total_notional < required_slice_notional:
            logger.warning(
                "Total order notional too small for %s: $%.4f < $%.4f",
                market,
                total_notional,
                required_slice_notional,
            )
            return [order.qty]

        target_slices = 1
        for candidate_slices in range(len(slices), 0, -1):
            slice_qty = order.qty / candidate_slices
            precise_slice_qty = self._precise_qty(market, slice_qty)
            if self._meets_notional(
                market,
                precise_slice_qty,
                ref_price,
                log_failure=False,
            ):
                target_slices = candidate_slices
                break

        if target_slices == len(slices):
            return slices

        if target_slices == 1:
            logger.info(
                "Reducing split for %s %s: %d -> 1 due to min notional",
                order.side,
                order.symbol,
                len(slices),
            )
            return [order.qty]

        slice_qty = order.qty / target_slices
        logger.info(
            "Reducing split for %s %s: %d -> %d due to min notional",
            order.side,
            order.symbol,
            len(slices),
            target_slices,
        )
        return [slice_qty] * target_slices

    def _best_price(self, market: str, side: str) -> float | None:
        try:
            orderbook = self._exchange.fetch_order_book(market, limit=5)
            if side == "buy":
                return orderbook["asks"][0][0] if orderbook["asks"] else None
            return orderbook["bids"][0][0] if orderbook["bids"] else None
        except Exception:
            return None

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
            self._managed_cash -= cost + fill.execution_cost
            self._managed_positions[fill.symbol] = (
                self._managed_positions.get(fill.symbol, 0.0) + fill.qty
            )
        elif fill.side == "sell":
            self._managed_cash += cost - fill.execution_cost
            self._managed_positions[fill.symbol] = (
                self._managed_positions.get(fill.symbol, 0.0) - fill.qty
            )

    def _extract_fee_cost(
        self,
        result: dict[str, Any],
        *,
        filled_qty: float,
        filled_price: float,
        market: str,
    ) -> CostEstimate:
        quote = market.split("/")[-1] if "/" in market else "USDT"

        def _convert(raw_cost: Any, currency: str | None) -> float | None:
            try:
                amount = float(raw_cost)
            except (TypeError, ValueError):
                return None
            curr = (currency or quote).upper()
            if curr == quote.upper():
                return amount
            if curr == market.split("/")[0].upper():
                return amount * filled_price
            return None

        fee = result.get("fee")
        if isinstance(fee, dict):
            converted = _convert(fee.get("cost"), fee.get("currency"))
            if converted is not None:
                return CostEstimate(commission=converted)

        fees = result.get("fees")
        if isinstance(fees, list):
            total = 0.0
            matched = False
            for item in fees:
                if not isinstance(item, dict):
                    continue
                converted = _convert(item.get("cost"), item.get("currency"))
                if converted is None:
                    continue
                total += converted
                matched = True
            if matched:
                return CostEstimate(commission=total)

        notional = filled_qty * filled_price
        return self._cost_model.estimate_order_cost(
            notional,
            include_modeled_slippage=False,
        )

    def submit_order(self, order: Order) -> Fill | None:
        if order.side == "sell" and not self.supports_short:
            current_qty = self.get_position(order.symbol)
            if order.qty > current_qty + 1e-12:
                logger.warning(
                    "Cannot sell %.6f %s with only %.6f managed in long-only mode",
                    order.qty, order.symbol, current_qty,
                )
                return None

        if self._optimizer is not None:
            max_attempts = max(1, int(getattr(self._optimizer, "max_deferral_attempts", 3)))
            sleep_seconds = max(0.0, float(getattr(self._optimizer, "deferral_sleep_seconds", 30.0)))
            for attempt in range(max_attempts):
                if self._optimizer.optimal_execution_window(order.side):
                    break
                logger.info(
                    "Order deferred by optimizer (attempt %d/%d): %s %s %.6f",
                    attempt + 1, max_attempts, order.side, order.symbol, order.qty,
                )
                if attempt + 1 < max_attempts:
                    time.sleep(sleep_seconds)
            else:
                logger.warning(
                    "Optimizer blocked after %d attempts, executing anyway: %s %s",
                    max_attempts, order.side, order.symbol,
                )

            slices = self._optimizer.split_order(order.qty)
            market = self._market_symbol(order.symbol)
            ref_price = self._best_price(market, order.side)
            if ref_price is not None:
                slices = self._adjust_slices_for_notional(order, slices, ref_price)
            if len(slices) > 1:
                fill = self._execute_slices(order, slices)
                if fill is not None:
                    self._update_managed_state(fill)
                return fill

        fill = self._submit_single(order)
        if fill is not None:
            self._update_managed_state(fill)
        return fill

    def constrain_intent(self, intent: ExecutionIntent) -> ConstraintResult:
        market = self._market_symbol(intent.symbol)
        result = apply_venue_constraints(
            intent,
            min_notional=self._min_notional(market),
            min_notional_buffer=self._min_notional_buffer,
            qty_rounder=lambda qty: self._precise_qty(market, qty),
        )
        if result.order is None and result.rejection_reason == "below_min_notional":
            required = (self._min_notional(market) or 0.0) * self._min_notional_buffer
            logger.warning(
                "Intent below min notional for %s: $%.4f < $%.4f (qty=%.8f, price=%.2f)",
                market,
                intent.notional_value,
                required,
                intent.qty,
                intent.reference_price,
            )
        elif result.order is None and result.rejection_reason == "rounded_to_zero":
            logger.warning(
                "Intent rounded to zero for %s after venue precision (qty=%.8f)",
                market,
                intent.qty,
            )
        return result

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

    def get_exchange_position(self, symbol: str) -> float:
        return self._exchange_position(symbol)

    def get_exchange_cash(self) -> float:
        return self._exchange_cash()

    def sync_reconciliation_baseline(self, symbols: list[str]) -> None:
        self._reconciliation_cash_offset = self._exchange_cash() - self._managed_cash
        offsets: dict[str, float] = {}
        for symbol in symbols:
            offsets[symbol] = (
                self._exchange_position(symbol)
                - self._managed_positions.get(symbol, 0.0)
            )
        self._reconciliation_position_offsets = offsets

    def get_reconciled_position(self, symbol: str) -> float:
        return (
            self._exchange_position(symbol)
            - self._reconciliation_position_offsets.get(symbol, 0.0)
        )

    def get_reconciled_cash(self) -> float:
        return self._exchange_cash() - self._reconciliation_cash_offset

    @property
    def supports_short(self) -> bool:
        return False

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
        market = self._market_symbol(symbol)
        base = market.split("/")[0]
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

        qty = self._precise_qty(market, order.qty)
        if qty <= 0:
            return None
        if not self._meets_notional(market, qty, best_ask):
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
            costs=self._extract_fee_cost(
                result,
                filled_qty=filled_qty,
                filled_price=filled_price,
                market=market,
            ),
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

        qty = self._precise_qty(market, order.qty)
        if qty <= 0:
            return None
        if not self._meets_notional(market, qty, best_bid):
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
            costs=self._extract_fee_cost(
                result,
                filled_qty=filled_qty,
                filled_price=filled_price,
                market=market,
            ),
        )
