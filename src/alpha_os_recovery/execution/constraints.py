"""Venue constraint helpers for turning intents into executable orders."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .planning import ExecutionIntent


@dataclass(frozen=True)
class ExecutableOrder:
    symbol: str
    side: str
    qty: float
    order_type: str = "market"
    reference_price: float = 0.0
    notional_value: float = 0.0


@dataclass(frozen=True)
class ConstraintResult:
    order: ExecutableOrder | None
    rejection_reason: str | None = None


def apply_venue_constraints(
    intent: ExecutionIntent,
    *,
    min_notional: float | None = None,
    min_notional_buffer: float = 1.0,
    qty_rounder: Callable[[float], float] | None = None,
    qty_epsilon: float = 1e-12,
) -> ConstraintResult:
    """Convert an intent into a venue-valid executable order."""
    qty = intent.qty
    if qty_rounder is not None:
        qty = float(qty_rounder(qty))
    if qty <= qty_epsilon:
        return ConstraintResult(order=None, rejection_reason="rounded_to_zero")

    notional_value = qty * intent.reference_price
    if min_notional is not None and min_notional > 0:
        required = min_notional * min_notional_buffer
        if notional_value < required:
            return ConstraintResult(order=None, rejection_reason="below_min_notional")

    return ConstraintResult(
        order=ExecutableOrder(
            symbol=intent.symbol,
            side=intent.side,
            qty=qty,
            order_type="market",
            reference_price=intent.reference_price,
            notional_value=notional_value,
        )
    )
