"""Runtime planning helpers shared by trade and replay paths."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetPosition:
    symbol: str
    qty: float
    reference_price: float
    dollar_target: float

    @property
    def notional_value(self) -> float:
        return abs(self.qty) * self.reference_price


@dataclass(frozen=True)
class ExecutionIntent:
    symbol: str
    side: str
    qty: float
    current_qty: float
    target_qty: float
    reference_price: float

    @property
    def delta_qty(self) -> float:
        return self.target_qty - self.current_qty

    @property
    def notional_value(self) -> float:
        return self.qty * self.reference_price


@dataclass(frozen=True)
class IntentDecision:
    intent: ExecutionIntent | None
    skip_reason: str | None = None


def build_target_position(
    *,
    symbol: str,
    adjusted_signal: float,
    portfolio_value: float,
    current_price: float,
    max_position_pct: float,
    min_trade_usd: float,
    supports_short: bool,
) -> TargetPosition:
    """Convert a normalized signal into a desired holding."""
    dollar_target = adjusted_signal * portfolio_value * max_position_pct
    qty = dollar_target / current_price if current_price > 0 else 0.0
    if abs(dollar_target) < min_trade_usd:
        qty = 0.0
    if not supports_short and qty < 0:
        qty = 0.0
    return TargetPosition(
        symbol=symbol,
        qty=float(qty),
        reference_price=float(current_price),
        dollar_target=float(dollar_target),
    )


def build_execution_intent(
    target: TargetPosition,
    *,
    current_qty: float,
    rebalance_deadband_usd: float = 0.0,
    qty_epsilon: float = 1e-6,
) -> ExecutionIntent | None:
    """Convert a target holding into an order intent."""
    return plan_execution_intent(
        target,
        current_qty=current_qty,
        rebalance_deadband_usd=rebalance_deadband_usd,
        qty_epsilon=qty_epsilon,
    ).intent


def plan_execution_intent(
    target: TargetPosition,
    *,
    current_qty: float,
    rebalance_deadband_usd: float = 0.0,
    qty_epsilon: float = 1e-6,
) -> IntentDecision:
    """Convert a target holding into an order intent with skip reason."""
    delta_qty = target.qty - current_qty
    if abs(delta_qty) < qty_epsilon:
        return IntentDecision(intent=None, skip_reason="no_delta")
    delta_notional = abs(delta_qty) * target.reference_price
    if delta_notional < rebalance_deadband_usd:
        return IntentDecision(intent=None, skip_reason="deadband")
    return IntentDecision(
        intent=ExecutionIntent(
            symbol=target.symbol,
            side="buy" if delta_qty > 0 else "sell",
            qty=abs(float(delta_qty)),
            current_qty=float(current_qty),
            target_qty=float(target.qty),
            reference_price=float(target.reference_price),
        )
    )
