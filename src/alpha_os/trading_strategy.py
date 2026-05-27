from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .evaluation_cost_config import TradingEnvironment
from .portfolio_construction_config import (
    PortfolioConstructionSpec,
)


def _normalize_optional(value: str | None) -> str | None:
    if value in {None, "", "-"}:
        return None
    return value


def _normalized_weight_caps(
    document: dict[str, Any],
    field_name: str,
) -> dict[str, float]:
    raw_value = document.get(field_name)
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError(f"risk_policy.{field_name} must be an object")
    normalized: dict[str, float] = {}
    for group_name, cap_value in raw_value.items():
        if not isinstance(group_name, str) or not group_name:
            raise ValueError(f"risk_policy.{field_name} keys must be non-empty strings")
        if not isinstance(cap_value, (int, float)):
            raise ValueError(
                f"risk_policy.{field_name}[{group_name}] must be numeric"
            )
        if float(cap_value) < 0.0:
            raise ValueError(
                f"risk_policy.{field_name}[{group_name}] must be >= 0"
            )
        normalized[str(group_name)] = float(cap_value)
    return normalized


def _serialize_weight_caps(weight_caps: dict[str, float] | None) -> str | None:
    if not weight_caps:
        return None
    return ",".join(
        f"{group_name}:{float(cap_value)}"
        for group_name, cap_value in sorted(weight_caps.items())
    )


def build_trading_strategy_id(
    *,
    subject_set_id: str | None,
    target_id: str | None,
    family_mix: str | None,
    sizing_method: str | None,
    sizing_engine: str | None,
    rebalance: str | None,
    long_only: bool | None,
    top_k: int | None,
    gross_exposure_cap: float | None,
    asset_class_weight_caps: dict[str, float] | None,
    cluster_weight_caps: dict[str, float] | None,
    direction_mode: str | None = None,
    target_vol: float | None = None,
    gross_leverage_cap: float | None = None,
    net_exposure_target: float | None = None,
) -> str:
    parts: list[tuple[str, str]] = []

    def add(name: str, value: object | None) -> None:
        if value in {None, "", "-"}:
            return
        parts.append((name, str(value)))

    add("subject_set", subject_set_id)
    add("target", target_id)
    add("family_mix", family_mix)
    add("sizing_method", sizing_method)
    add("sizing_engine", sizing_engine)
    add("rebalance", rebalance)
    add("direction_mode", direction_mode)
    add("top_k", top_k)
    add("gross_exposure_cap", gross_exposure_cap)
    add("asset_class_weight_caps", _serialize_weight_caps(asset_class_weight_caps))
    add("cluster_weight_caps", _serialize_weight_caps(cluster_weight_caps))
    add("target_vol", target_vol)
    add("gross_leverage_cap", gross_leverage_cap)
    add("net_exposure_target", net_exposure_target)
    payload = "|".join(f"{name}={value}" for name, value in sorted(parts))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"strategy:{digest}"


def _rebalance_interval_steps_from_document(document: dict[str, Any]) -> int:
    raw_steps = document.get("rebalance_interval_steps")
    if raw_steps is None:
        raise ValueError("trading strategy rebalance_interval_steps is required")
    steps = int(raw_steps)
    if steps < 1:
        raise ValueError("rebalance_interval_steps must be >= 1")
    return steps


def _rebalance_label(rebalance_interval_steps: int) -> str:
    return f"every_{int(rebalance_interval_steps)}_steps"


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
    position_rule_id: str
    family_mix: str | None
    portfolio_construction: PortfolioConstructionSpec
    trading_environment: TradingEnvironment
    created_at: str
    top_k: int | None = None
    rebalance_interval_steps: int = 1

    def __post_init__(self) -> None:
        top_k = self.top_k
        construction_top_k = getattr(self.portfolio_construction, "top_k", None)
        if top_k is None and construction_top_k is not None:
            top_k = int(construction_top_k)
        if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
            raise ValueError("trading strategy top_k must be >= 1")
        object.__setattr__(self, "top_k", top_k)
        if (
            not isinstance(self.rebalance_interval_steps, int)
            or self.rebalance_interval_steps < 1
        ):
            raise ValueError("trading strategy rebalance_interval_steps must be >= 1")
        if (
            self.portfolio_construction.rebalance_interval_steps
            != self.rebalance_interval_steps
        ):
            object.__setattr__(
                self,
                "portfolio_construction",
                replace(
                    self.portfolio_construction,
                    rebalance_interval_steps=self.rebalance_interval_steps,
                ),
            )

    def to_document(self) -> dict[str, Any]:
        document = {
            "strategy_id": self.strategy_id,
            "label": self.label,
            "subject_set_id": self.subject_set_id,
            "target_id": self.target_id,
            "position_rule_id": self.position_rule_id,
            "family_mix": self.family_mix,
            "portfolio_construction": self.portfolio_construction.to_document(),
            "trading_environment": self.trading_environment.to_document(),
            "rebalance_interval_steps": self.rebalance_interval_steps,
            "created_at": self.created_at,
        }
        if self.top_k is not None:
            document["top_k"] = self.top_k
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "TradingStrategySpec":
        top_k = document.get("top_k")
        portfolio_construction = PortfolioConstructionSpec.from_document(
            document.get("portfolio_construction")
        )
        rebalance_interval_steps = _rebalance_interval_steps_from_document(document)
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
            position_rule_id=str(document.get("position_rule_id", "constant_hold")),
            family_mix=_normalize_optional(
                None if document.get("family_mix") is None else str(document["family_mix"])
            ),
            portfolio_construction=portfolio_construction,
            trading_environment=TradingEnvironment.from_document(
                document.get("trading_environment")
            ),
            top_k=None if top_k is None else int(top_k),
            rebalance_interval_steps=int(rebalance_interval_steps),
            created_at=str(document["created_at"]),
        )
