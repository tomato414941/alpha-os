from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _float_from_document(
    document: dict[str, Any],
    field_name: str,
    *,
    default: float,
) -> float:
    value = document.get(field_name)
    return default if value is None else float(value)


@dataclass(frozen=True)
class EvaluationRebalanceFrictionPolicySpec:
    turnover_friction: float = 0.0
    no_trade_band: float = 0.0
    execution_cost_aversion: float = 1.0
    turnover_budget: float | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("turnover_friction", self.turnover_friction),
            ("no_trade_band", self.no_trade_band),
            ("execution_cost_aversion", self.execution_cost_aversion),
        ):
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"rebalance_friction_policy.{field_name} must be numeric"
                )
            if float(value) < 0.0:
                raise ValueError(
                    f"rebalance_friction_policy.{field_name} must be >= 0"
                )
        if self.turnover_budget is not None and not isinstance(
            self.turnover_budget, (int, float)
        ):
            raise ValueError("rebalance_friction_policy.turnover_budget must be numeric")
        if self.turnover_budget is not None and float(self.turnover_budget) < 0.0:
            raise ValueError("rebalance_friction_policy.turnover_budget must be >= 0")

    def to_document(self) -> dict[str, Any]:
        return {
            "turnover_friction": self.turnover_friction,
            "no_trade_band": self.no_trade_band,
            "execution_cost_aversion": self.execution_cost_aversion,
            "turnover_budget": self.turnover_budget,
        }

    @classmethod
    def from_document(
        cls, document: dict[str, Any] | None
    ) -> "EvaluationRebalanceFrictionPolicySpec":
        if document is None:
            return cls()
        if not isinstance(document, dict):
            raise ValueError("rebalance_friction_policy must be an object")
        return cls(
            turnover_friction=float(document.get("turnover_friction", 0.0)),
            no_trade_band=float(document.get("no_trade_band", 0.0)),
            execution_cost_aversion=float(
                document.get("execution_cost_aversion", 1.0)
            ),
            turnover_budget=(
                None
                if document.get("turnover_budget") is None
                else float(document.get("turnover_budget"))
            ),
        )


@dataclass(frozen=True)
class ExecutionCostAssumptionsSpec:
    market_impact_bps: float = 0.0
    fee_bps: float = 0.0
    bid_ask_spread_bps: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.market_impact_bps, (int, float)):
            raise ValueError(
                "execution_cost_assumptions.market_impact_bps must be numeric"
            )
        if not isinstance(self.fee_bps, (int, float)):
            raise ValueError("execution_cost_assumptions.fee_bps must be numeric")
        if not isinstance(self.bid_ask_spread_bps, (int, float)):
            raise ValueError("execution_cost_assumptions.bid_ask_spread_bps must be numeric")

    def to_document(self) -> dict[str, Any]:
        return {
            "market_impact_bps": self.market_impact_bps,
            "fee_bps": self.fee_bps,
            "bid_ask_spread_bps": self.bid_ask_spread_bps,
        }

    @classmethod
    def from_document(
        cls, document: dict[str, Any] | None
    ) -> "ExecutionCostAssumptionsSpec":
        if document is None:
            return cls()
        if not isinstance(document, dict):
            raise ValueError("execution_cost_assumptions must be an object")
        return cls(
            market_impact_bps=float(document.get("market_impact_bps", 0.0)),
            fee_bps=float(document.get("fee_bps", 0.0)),
            bid_ask_spread_bps=float(document.get("bid_ask_spread_bps", 0.0)),
        )


@dataclass(frozen=True)
class HoldingCostAssumptionsSpec:
    funding_bps_per_step: float = 0.0
    borrow_fee_bps_per_step: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.funding_bps_per_step, (int, float)):
            raise ValueError(
                "holding_cost_assumptions.funding_bps_per_step must be numeric"
            )
        if not isinstance(self.borrow_fee_bps_per_step, (int, float)):
            raise ValueError(
                "holding_cost_assumptions.borrow_fee_bps_per_step must be numeric"
            )

    def to_document(self) -> dict[str, Any]:
        return {
            "funding_bps_per_step": self.funding_bps_per_step,
            "borrow_fee_bps_per_step": self.borrow_fee_bps_per_step,
        }

    @classmethod
    def from_document(
        cls, document: dict[str, Any] | None
    ) -> "HoldingCostAssumptionsSpec":
        if document is None:
            return cls()
        if not isinstance(document, dict):
            raise ValueError("holding_cost_assumptions must be an object")
        return cls(
            funding_bps_per_step=float(document.get("funding_bps_per_step", 0.0)),
            borrow_fee_bps_per_step=float(
                document.get("borrow_fee_bps_per_step", 0.0)
            ),
        )


@dataclass(frozen=True)
class TradingEnvironment:
    market_impact_bps: float = 0.0
    fee_bps: float = 0.0
    bid_ask_spread_bps: float = 0.0
    funding_bps_per_step: float = 0.0
    borrow_fee_bps_per_step: float = 0.0

    def __post_init__(self) -> None:
        for field_name, value in (
            ("market_impact_bps", self.market_impact_bps),
            ("fee_bps", self.fee_bps),
            ("bid_ask_spread_bps", self.bid_ask_spread_bps),
            ("funding_bps_per_step", self.funding_bps_per_step),
            ("borrow_fee_bps_per_step", self.borrow_fee_bps_per_step),
        ):
            if not isinstance(value, (int, float)):
                raise ValueError(f"trading environment {field_name} must be numeric")

    @classmethod
    def from_document(cls, document: dict[str, Any] | None) -> "TradingEnvironment":
        if document is None:
            return cls()
        if not isinstance(document, dict):
            raise ValueError("trading_environment must be an object")
        return cls(
            market_impact_bps=_float_from_document(
                document, "market_impact_bps", default=0.0
            ),
            fee_bps=_float_from_document(document, "fee_bps", default=0.0),
            bid_ask_spread_bps=_float_from_document(
                document, "bid_ask_spread_bps", default=0.0
            ),
            funding_bps_per_step=_float_from_document(
                document, "funding_bps_per_step", default=0.0
            ),
            borrow_fee_bps_per_step=_float_from_document(
                document, "borrow_fee_bps_per_step", default=0.0
            ),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "market_impact_bps": self.market_impact_bps,
            "fee_bps": self.fee_bps,
            "bid_ask_spread_bps": self.bid_ask_spread_bps,
            "funding_bps_per_step": self.funding_bps_per_step,
            "borrow_fee_bps_per_step": self.borrow_fee_bps_per_step,
        }
