"""Execution cost estimates shared by paper, replay, and live executors."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostEstimate:
    commission: float = 0.0
    modeled_slippage: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.commission + self.modeled_slippage


@dataclass(frozen=True)
class ExecutionCostModel:
    commission_pct: float = 0.10
    modeled_slippage_pct: float = 0.05

    def estimate_order_cost(
        self,
        notional: float,
        *,
        include_modeled_slippage: bool,
    ) -> CostEstimate:
        base = abs(notional)
        commission = base * self.commission_pct / 100.0
        modeled_slippage = 0.0
        if include_modeled_slippage:
            modeled_slippage = base * self.modeled_slippage_pct / 100.0
        return CostEstimate(
            commission=commission,
            modeled_slippage=modeled_slippage,
        )
