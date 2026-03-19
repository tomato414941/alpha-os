"""Trade executor — abstract interface + Alpaca integration."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .constraints import ConstraintResult, apply_venue_constraints
from .costs import CostEstimate
from .planning import ExecutionIntent

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
    costs: CostEstimate = field(default_factory=CostEstimate)

    @property
    def execution_cost(self) -> float:
        return self.costs.total_cost


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

    def get_exchange_position(self, symbol: str) -> float:
        """Exchange-level position. Defaults to managed/in-memory position."""
        return self.get_position(symbol)

    def get_exchange_cash(self) -> float:
        """Exchange-level cash. Defaults to managed/in-memory cash."""
        return self.get_cash()

    def sync_reconciliation_baseline(self, symbols: list[str]) -> None:
        """Refresh any executor-specific reconciliation baseline."""
        return None

    def get_reconciled_position(self, symbol: str) -> float:
        """Position value to compare against internal managed state."""
        return self.get_exchange_position(symbol)

    def get_reconciled_cash(self) -> float:
        """Cash value to compare against internal managed state."""
        return self.get_exchange_cash()

    def set_price(self, symbol: str, price: float) -> None:
        """Set current price for a symbol. No-op for live executors."""
        pass

    def fetch_ticker_price(self, symbol: str) -> float | None:
        """Fetch real-time price from exchange. Returns None if unavailable."""
        return None

    @property
    def supports_short(self) -> bool:
        """Whether the executor can maintain net short positions."""
        return False

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

    def constrain_intent(self, intent: ExecutionIntent) -> ConstraintResult:
        """Convert an execution intent into a venue-valid order."""
        return apply_venue_constraints(intent)

    def execute_intent(self, intent: ExecutionIntent) -> Fill | None:
        """Execute an intent after applying venue constraints."""
        result = self.constrain_intent(intent)
        if result.order is None:
            return None
        order = Order(
            symbol=result.order.symbol,
            side=result.order.side,
            qty=result.order.qty,
            order_type=result.order.order_type,
        )
        return self.submit_order(order)

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


