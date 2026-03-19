"""Polymarket CLOB executor — limit orders on prediction markets."""
from __future__ import annotations

import logging
import time

from .costs import PolymarketCostModel
from .executor import Executor, Order, Fill
from .secrets import get_env_or_secret, load_secrets

logger = logging.getLogger(__name__)

_SECRETS_FILE = "polymarket"

# Module-level alias for test compatibility
_load_secrets = load_secrets


def _get_credentials() -> tuple[str, str]:
    """Return (private_key, api_key) from env vars or ~/.secrets/polymarket."""
    creds = get_env_or_secret(
        ["POLYMARKET_PRIVATE_KEY", "POLYMARKET_API_KEY"],
        _SECRETS_FILE,
    )
    return creds.get("POLYMARKET_PRIVATE_KEY", ""), creds.get("POLYMARKET_API_KEY", "")


def _create_clob_client(host: str = "https://clob.polymarket.com"):
    """Create authenticated Polymarket CLOB client."""
    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        raise ImportError(
            "py-clob-client is required for Polymarket trading. "
            "Install with: pip install 'alpha-os[polymarket]'"
        )

    private_key, api_key = _get_credentials()
    if not private_key:
        raise ValueError(
            "Polymarket credentials not found. "
            "Set POLYMARKET_PRIVATE_KEY/POLYMARKET_API_KEY env vars "
            "or create ~/.secrets/polymarket"
        )

    chain_id = 137  # Polygon mainnet
    client = ClobClient(host, key=private_key, chain_id=chain_id)

    if api_key:
        client.set_api_creds(client.create_or_derive_api_creds())
    else:
        client.set_api_creds(client.create_or_derive_api_creds())

    logger.info("Created Polymarket CLOB client (host=%s)", host)
    return client


class PolymarketExecutor(Executor):
    """Polymarket prediction market executor with managed position tracking.

    Uses limit orders via the CLOB API. Tracks positions created by
    alpha-os trades only, separate from the full Polymarket account.

    Binary outcome markets: price in [0, 1], qty in shares.
    Buy YES = go long on outcome, Buy NO = go short on outcome.
    """

    def __init__(
        self,
        host: str = "https://clob.polymarket.com",
        max_position_per_market_usd: float = 100.0,
        initial_capital: float = 1000.0,
        cost_model: PolymarketCostModel | None = None,
    ) -> None:
        self._client = _create_clob_client(host=host)
        self._host = host
        self._max_position_per_market = max_position_per_market_usd
        self._cost_model = cost_model or PolymarketCostModel()
        self._managed_cash: float = initial_capital
        self._managed_positions: dict[str, float] = {}
        self._initial_capital = initial_capital
        self._fills: list[Fill] = []

    @staticmethod
    def _clob_side(side: str) -> str:
        """Map order side to CLOB constant string."""
        try:
            from py_clob_client.order_builder.constants import BUY, SELL
            return BUY if side == "buy" else SELL
        except ImportError:
            return "BUY" if side == "buy" else "SELL"

    def submit_order(self, order: Order) -> Fill | None:
        """Submit a limit order to Polymarket CLOB.

        Order.symbol is the token_id (condition_id) of the market.
        Order.side is "buy" or "sell" (for YES tokens).
        """
        if order.side == "sell" and not self.supports_short:
            current_qty = self.get_position(order.symbol)
            if order.qty > current_qty + 1e-12:
                logger.warning(
                    "Cannot sell %.6f %s with only %.6f shares",
                    order.qty, order.symbol, current_qty,
                )
                return None

        clob_side = self._clob_side(order.side)

        price = self._get_market_price(order.symbol)
        if price is None:
            logger.error("Cannot get price for market %s", order.symbol)
            return None

        notional = order.qty * price
        if notional > self._max_position_per_market:
            logger.warning(
                "Order notional $%.2f exceeds max $%.2f for %s",
                notional, self._max_position_per_market, order.symbol,
            )
            return None

        t0 = time.perf_counter()
        try:
            signed_order = self._client.create_and_post_order(
                {
                    "token_id": order.symbol,
                    "price": price,
                    "size": order.qty,
                    "side": clob_side,
                }
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            order_id = ""
            if isinstance(signed_order, dict):
                order_id = str(signed_order.get("orderID", signed_order.get("id", "")))

            costs = self._cost_model.estimate_order_cost(notional, is_maker=True)

            fill = Fill(
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                price=price,
                order_id=order_id,
                slippage_bps=0.0,
                latency_ms=latency_ms,
                costs=costs,
            )
            self._update_managed_state(fill)
            self._fills.append(fill)
            return fill

        except Exception as e:
            logger.error("Polymarket order failed for %s: %s", order.symbol, e)
            return None

    def _get_market_price(self, token_id: str) -> float | None:
        """Get current market price for a token."""
        try:
            price_data = self._client.get_last_trade_price(token_id)
            if isinstance(price_data, dict):
                return float(price_data.get("price", 0))
            return float(price_data)
        except Exception as e:
            logger.warning("Failed to get price for %s: %s", token_id, e)
            return None

    def _update_managed_state(self, fill: Fill) -> None:
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

    def get_position(self, symbol: str) -> float:
        return self._managed_positions.get(symbol, 0.0)

    def get_cash(self) -> float:
        return self._managed_cash

    def fetch_ticker_price(self, symbol: str) -> float | None:
        return self._get_market_price(symbol)

    @property
    def supports_short(self) -> bool:
        return False

    @property
    def portfolio_value(self) -> float:
        total = self._managed_cash
        for symbol, qty in self._managed_positions.items():
            if abs(qty) > 1e-8:
                price = self._get_market_price(symbol)
                if price is not None:
                    total += qty * price
        return total

    @property
    def all_positions(self) -> dict[str, float]:
        return {s: q for s, q in self._managed_positions.items() if abs(q) > 1e-8}

    @property
    def all_fills(self) -> list[Fill]:
        return list(self._fills)

    def settle_market(self, token_id: str) -> float | None:
        """Handle market settlement. Returns payout or None on failure.

        Binary outcome: payout = qty if outcome is YES, 0 if NO.
        """
        qty = self._managed_positions.get(token_id, 0.0)
        if abs(qty) < 1e-8:
            return 0.0

        try:
            market = self._client.get_market(token_id)
            if not isinstance(market, dict):
                return None

            resolved = market.get("resolved", False)
            if not resolved:
                logger.info("Market %s not yet resolved", token_id)
                return None

            outcome_price = float(market.get("outcome_price", 0))
            payout = qty * outcome_price
            self._managed_cash += payout
            self._managed_positions.pop(token_id, None)
            logger.info(
                "Settled market %s: qty=%.4f, outcome_price=%.4f, payout=$%.2f",
                token_id, qty, outcome_price, payout,
            )
            return payout

        except Exception as e:
            logger.error("Failed to settle market %s: %s", token_id, e)
            return None
