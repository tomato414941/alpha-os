"""Paper trading executor — simulates order execution locally."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .executor import Executor, Order, Fill

logger = logging.getLogger(__name__)


class PaperExecutor(Executor):
    """In-memory paper trading executor for backtesting and simulation."""

    def __init__(self, initial_cash: float = 10000.0):
        self._cash = initial_cash
        self._positions: dict[str, float] = {}
        self._prices: dict[str, float] = {}
        self._fills: list[Fill] = []
        self._order_counter = 0

    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = price

    def set_prices(self, prices: dict[str, float]) -> None:
        self._prices.update(prices)

    def submit_order(self, order: Order) -> Fill | None:
        price = self._prices.get(order.symbol)
        if price is None or price <= 0:
            logger.warning(f"No price for {order.symbol}, skipping order")
            return None

        cost = order.qty * price
        if order.side == "buy":
            if cost > self._cash:
                logger.warning(f"Insufficient cash: need {cost:.2f}, have {self._cash:.2f}")
                return None
            self._cash -= cost
            self._positions[order.symbol] = self._positions.get(order.symbol, 0) + order.qty
        elif order.side == "sell":
            self._cash += cost
            self._positions[order.symbol] = self._positions.get(order.symbol, 0) - order.qty
        else:
            return None

        self._order_counter += 1
        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=price,
            order_id=f"paper-{self._order_counter}",
        )
        self._fills.append(fill)
        return fill

    def get_position(self, symbol: str) -> float:
        return self._positions.get(symbol, 0.0)

    def get_cash(self) -> float:
        return self._cash

    @property
    def portfolio_value(self) -> float:
        positions_value = sum(
            qty * self._prices.get(sym, 0)
            for sym, qty in self._positions.items()
        )
        return self._cash + positions_value

    @property
    def all_fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def all_positions(self) -> dict[str, float]:
        return dict(self._positions)
