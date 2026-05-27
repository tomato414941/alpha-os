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
class TradingEnvironment:
    turnover_cost_rate: float = 0.0
    market_impact_bps: float = 0.0
    fee_bps: float = 0.0
    bid_ask_spread_bps: float = 0.0
    funding_bps_per_step: float = 0.0
    borrow_fee_bps_per_step: float = 0.0

    def __post_init__(self) -> None:
        for field_name, value in (
            ("turnover_cost_rate", self.turnover_cost_rate),
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
            turnover_cost_rate=_float_from_document(
                document, "turnover_cost_rate", default=0.0
            ),
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
            "turnover_cost_rate": self.turnover_cost_rate,
            "market_impact_bps": self.market_impact_bps,
            "fee_bps": self.fee_bps,
            "bid_ask_spread_bps": self.bid_ask_spread_bps,
            "funding_bps_per_step": self.funding_bps_per_step,
            "borrow_fee_bps_per_step": self.borrow_fee_bps_per_step,
        }
