from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .portfolio_construction_config import (
    PortfolioConstructionSpec,
)


def _normalize_optional(value: str | None) -> str | None:
    if value in {None, "", "-"}:
        return None
    return value


@dataclass(frozen=True)
class TradingStrategyInput:
    pass


@dataclass(frozen=True)
class TradingStrategyOutput:
    pass


class TradingStrategy(Protocol):
    """Black-box policy contract for trading decision components."""

    def decide(self, strategy_input: TradingStrategyInput) -> TradingStrategyOutput:
        ...


# Persisted strategy records are not the trading strategy contract above.
# Keep this class narrow and avoid turning it into a base schema for all strategy
# implementations.
@dataclass(frozen=True)
class TradingStrategySpec:
    strategy_id: str
    label: str
    subject_set_id: str | None
    target_id: str | None
    portfolio_construction: PortfolioConstructionSpec
    created_at: str

    def to_document(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "label": self.label,
            "subject_set_id": self.subject_set_id,
            "target_id": self.target_id,
            "portfolio_construction": self.portfolio_construction.to_document(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "TradingStrategySpec":
        portfolio_construction = PortfolioConstructionSpec.from_document(
            document.get("portfolio_construction")
        )
        return cls(
            strategy_id=str(document["strategy_id"]),
            label=str(document["label"]),
            subject_set_id=_normalize_optional(
                None
                if document.get("subject_set_id") is None
                else str(document["subject_set_id"])
            ),
            target_id=_normalize_optional(
                None if document.get("target_id") is None else str(document["target_id"])
            ),
            portfolio_construction=portfolio_construction,
            created_at=str(document["created_at"]),
        )
