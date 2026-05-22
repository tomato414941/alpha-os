from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _optional_bool_from_document(
    document: dict[str, Any],
    field_name: str,
    *,
    default: bool,
) -> bool:
    value = document.get(field_name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", "", "-"}:
            return False
    raise ValueError(f"{field_name} must be boolean")


@dataclass(frozen=True)
class EvaluationRebalanceFrictionPolicySpec:
    execution_mode: str = "utility_priority"
    turnover_friction: float = 0.0
    no_trade_band: float = 0.0
    execution_cost_aversion: float = 1.0
    turnover_budget: float | None = None
    benefit_scale: float = 1.0
    min_trade_utility: float = 0.0
    uncertainty_aversion: float = 1.0
    risk_aversion: float = 0.0
    partial_fill_enabled: bool = True

    def __post_init__(self) -> None:
        if self.execution_mode not in {"threshold", "utility_priority"}:
            raise ValueError(
                "rebalance_friction_policy.execution_mode must be threshold "
                "or utility_priority"
            )
        for field_name, value in (
            ("turnover_friction", self.turnover_friction),
            ("no_trade_band", self.no_trade_band),
            ("execution_cost_aversion", self.execution_cost_aversion),
            ("benefit_scale", self.benefit_scale),
            ("min_trade_utility", self.min_trade_utility),
            ("uncertainty_aversion", self.uncertainty_aversion),
            ("risk_aversion", self.risk_aversion),
        ):
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"rebalance_friction_policy.{field_name} must be numeric"
                )
            if float(value) < 0.0:
                raise ValueError(
                    f"rebalance_friction_policy.{field_name} must be >= 0"
                )
        if not isinstance(self.partial_fill_enabled, bool):
            raise ValueError(
                "rebalance_friction_policy.partial_fill_enabled must be boolean"
            )
        if self.turnover_budget is not None and not isinstance(
            self.turnover_budget, (int, float)
        ):
            raise ValueError("rebalance_friction_policy.turnover_budget must be numeric")
        if self.turnover_budget is not None and float(self.turnover_budget) < 0.0:
            raise ValueError("rebalance_friction_policy.turnover_budget must be >= 0")

    def to_document(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "turnover_friction": self.turnover_friction,
            "no_trade_band": self.no_trade_band,
            "execution_cost_aversion": self.execution_cost_aversion,
            "turnover_budget": self.turnover_budget,
            "benefit_scale": self.benefit_scale,
            "min_trade_utility": self.min_trade_utility,
            "uncertainty_aversion": self.uncertainty_aversion,
            "risk_aversion": self.risk_aversion,
            "partial_fill_enabled": self.partial_fill_enabled,
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
            execution_mode=str(document.get("execution_mode", "utility_priority")),
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
            benefit_scale=float(document.get("benefit_scale", 1.0)),
            min_trade_utility=float(document.get("min_trade_utility", 0.0)),
            uncertainty_aversion=float(document.get("uncertainty_aversion", 1.0)),
            risk_aversion=float(document.get("risk_aversion", 0.0)),
            partial_fill_enabled=_optional_bool_from_document(
                document,
                "partial_fill_enabled",
                default=True,
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
    def from_cost_assumptions(
        cls,
        *,
        execution_cost_assumptions: ExecutionCostAssumptionsSpec,
        holding_cost_assumptions: HoldingCostAssumptionsSpec,
    ) -> "TradingEnvironment":
        return cls(
            market_impact_bps=float(execution_cost_assumptions.market_impact_bps),
            fee_bps=float(execution_cost_assumptions.fee_bps),
            bid_ask_spread_bps=float(execution_cost_assumptions.bid_ask_spread_bps),
            funding_bps_per_step=float(holding_cost_assumptions.funding_bps_per_step),
            borrow_fee_bps_per_step=float(
                holding_cost_assumptions.borrow_fee_bps_per_step
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
