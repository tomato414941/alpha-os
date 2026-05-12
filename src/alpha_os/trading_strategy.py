from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any

from .portfolio_construction_config import (
    PortfolioConstructionSpec,
)
from .strategy_sleeves import StrategySleeveCompositionSpec


def _normalize_optional(value: str | None) -> str | None:
    if value in {None, "", "-"}:
        return None
    return value


def _optional_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", "", "-"}:
            return False
    raise ValueError(f"{field_name} must be boolean")


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
    signal_discovery_id: str | None,
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
    market_impact_bps: float | None = None,
    fee_bps: float | None = None,
    bid_ask_spread_bps: float | None = None,
    turnover_friction: float | None = None,
    no_trade_band: float | None = None,
    funding_bps_per_step: float | None = None,
    borrow_fee_bps_per_step: float | None = None,
    adaptation_enabled: bool | None = None,
    adaptation_blend: float | None = None,
    sleeve_composition: StrategySleeveCompositionSpec | None = None,
) -> str:
    parts: list[tuple[str, str]] = []

    def add(name: str, value: object | None) -> None:
        if value in {None, "", "-"}:
            return
        parts.append((name, str(value)))

    add("signal_discovery", signal_discovery_id)
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
    add("market_impact_bps", market_impact_bps)
    add("fee_bps", fee_bps)
    add("bid_ask_spread_bps", bid_ask_spread_bps)
    add("funding_bps_per_step", funding_bps_per_step)
    add("borrow_fee_bps_per_step", borrow_fee_bps_per_step)
    add("turnover_friction", turnover_friction)
    add("no_trade_band", no_trade_band)
    add(
        "adaptation_enabled",
        None if adaptation_enabled is None else str(adaptation_enabled).lower(),
    )
    add("adaptation_blend", adaptation_blend)
    if sleeve_composition is not None:
        add("sleeve_composition", sleeve_composition.stable_payload())

    payload = "|".join(f"{name}={value}" for name, value in sorted(parts))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"strategy:{digest}"


@dataclass(frozen=True)
class TradingStrategyScopeSpec:
    subject_set_id: str | None
    target_id: str | None

    def to_document(self) -> dict[str, Any]:
        return {
            "subject_set_id": self.subject_set_id,
            "target_id": self.target_id,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "TradingStrategyScopeSpec":
        return cls(
            subject_set_id=_normalize_optional(
                None
                if document.get("subject_set_id") is None
                else str(document["subject_set_id"])
            ),
            target_id=_normalize_optional(
                None if document.get("target_id") is None else str(document["target_id"])
            ),
        )


@dataclass(frozen=True)
class RebalanceFrictionPolicySpec:
    turnover_friction: float | None
    no_trade_band: float | None
    execution_cost_aversion: float | None = None
    execution_mode: str | None = None
    turnover_budget: float | None = None
    benefit_scale: float | None = None
    min_trade_utility: float | None = None
    uncertainty_aversion: float | None = None
    risk_aversion: float | None = None
    partial_fill_enabled: bool | None = None

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
    def from_document(cls, document: dict[str, Any]) -> "RebalanceFrictionPolicySpec":
        turnover_friction = document.get("turnover_friction")
        no_trade_band = document.get("no_trade_band")
        execution_cost_aversion = document.get("execution_cost_aversion")
        execution_mode = document.get("execution_mode")
        turnover_budget = document.get("turnover_budget")
        benefit_scale = document.get("benefit_scale")
        min_trade_utility = document.get("min_trade_utility")
        uncertainty_aversion = document.get("uncertainty_aversion")
        risk_aversion = document.get("risk_aversion")
        partial_fill_enabled = document.get("partial_fill_enabled")
        return cls(
            turnover_friction=(
                None if turnover_friction is None else float(turnover_friction)
            ),
            no_trade_band=None if no_trade_band is None else float(no_trade_band),
            execution_cost_aversion=(
                None
                if execution_cost_aversion is None
                else float(execution_cost_aversion)
            ),
            execution_mode=(
                None if execution_mode is None else str(execution_mode)
            ),
            turnover_budget=(
                None if turnover_budget is None else float(turnover_budget)
            ),
            benefit_scale=None if benefit_scale is None else float(benefit_scale),
            min_trade_utility=(
                None if min_trade_utility is None else float(min_trade_utility)
            ),
            uncertainty_aversion=(
                None if uncertainty_aversion is None else float(uncertainty_aversion)
            ),
            risk_aversion=None if risk_aversion is None else float(risk_aversion),
            partial_fill_enabled=_optional_bool(
                partial_fill_enabled,
                field_name="rebalance_friction_policy.partial_fill_enabled",
            ),
        )


@dataclass(frozen=True)
class ExecutionPolicySpec:
    market_impact_bps: float | None
    fee_bps: float | None = None
    bid_ask_spread_bps: float | None = None

    def to_document(self) -> dict[str, Any]:
        return {
            "market_impact_bps": self.market_impact_bps,
            "fee_bps": self.fee_bps,
            "bid_ask_spread_bps": self.bid_ask_spread_bps,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "ExecutionPolicySpec":
        market_impact_bps = document.get("market_impact_bps")
        fee_bps = document.get("fee_bps")
        bid_ask_spread_bps = document.get("bid_ask_spread_bps")
        return cls(
            market_impact_bps=(
                None
                if market_impact_bps is None
                else float(market_impact_bps)
            ),
            fee_bps=None if fee_bps is None else float(fee_bps),
            bid_ask_spread_bps=None if bid_ask_spread_bps is None else float(bid_ask_spread_bps),
        )


@dataclass(frozen=True)
class HoldingCostPolicySpec:
    funding_bps_per_step: float | None = None
    borrow_fee_bps_per_step: float | None = None

    def to_document(self) -> dict[str, Any]:
        return {
            "funding_bps_per_step": self.funding_bps_per_step,
            "borrow_fee_bps_per_step": self.borrow_fee_bps_per_step,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "HoldingCostPolicySpec":
        funding_bps_per_step = document.get("funding_bps_per_step")
        borrow_fee_bps_per_step = document.get("borrow_fee_bps_per_step")
        return cls(
            funding_bps_per_step=(
                None
                if funding_bps_per_step is None
                else float(funding_bps_per_step)
            ),
            borrow_fee_bps_per_step=(
                None
                if borrow_fee_bps_per_step is None
                else float(borrow_fee_bps_per_step)
            ),
        )


@dataclass(frozen=True)
class AdaptationPolicySpec:
    enabled: bool
    adaptation_blend: float

    def to_document(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "adaptation_blend": self.adaptation_blend,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "AdaptationPolicySpec":
        enabled = document.get("enabled", False)
        if isinstance(enabled, str):
            normalized_enabled = _normalize_optional(enabled)
            if normalized_enabled is None:
                enabled = False
            elif normalized_enabled == "true":
                enabled = True
            elif normalized_enabled == "false":
                enabled = False
            else:
                raise ValueError(f"unsupported adaptation enabled value: {enabled}")
        adaptation_blend = document.get("adaptation_blend", 0.2)
        return cls(
            enabled=bool(enabled),
            adaptation_blend=float(adaptation_blend),
        )


def _rebalance_interval_steps_from_document(document: dict[str, Any]) -> int:
    raw_steps = document.get("rebalance_interval_steps")
    if raw_steps is None:
        raise ValueError("strategy portfolio rebalance_interval_steps is required")
    steps = int(raw_steps)
    if steps < 1:
        raise ValueError("rebalance_interval_steps must be >= 1")
    return steps


def _rebalance_label(rebalance_interval_steps: int) -> str:
    return f"every_{int(rebalance_interval_steps)}_steps"


@dataclass(frozen=True)
class StrategyPortfolioSpec:
    portfolio_construction: PortfolioConstructionSpec
    rebalance_friction_policy: RebalanceFrictionPolicySpec
    execution_policy: ExecutionPolicySpec
    holding_cost_policy: HoldingCostPolicySpec = field(
        default_factory=HoldingCostPolicySpec
    )
    selection_kind: str = "all_assets"
    top_k: int | None = None
    rebalance_interval_steps: int = 1

    def __post_init__(self) -> None:
        top_k = self.top_k
        construction_top_k = getattr(self.portfolio_construction, "top_k", None)
        if top_k is None and construction_top_k is not None:
            top_k = int(construction_top_k)
        if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
            raise ValueError("strategy portfolio top_k must be >= 1")
        object.__setattr__(self, "top_k", top_k)
        if (
            not isinstance(self.rebalance_interval_steps, int)
            or self.rebalance_interval_steps < 1
        ):
            raise ValueError("strategy portfolio rebalance_interval_steps must be >= 1")
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
            "portfolio_construction": self.portfolio_construction.to_document(),
            "rebalance_friction_policy": self.rebalance_friction_policy.to_document(),
            "execution_policy": self.execution_policy.to_document(),
            "rebalance_interval_steps": self.rebalance_interval_steps,
        }
        if (
            self.holding_cost_policy.funding_bps_per_step is not None
            or self.holding_cost_policy.borrow_fee_bps_per_step is not None
        ):
            document["holding_cost_policy"] = self.holding_cost_policy.to_document()
        if self.selection_kind != "all_assets":
            document["selection_kind"] = self.selection_kind
        if self.top_k is not None:
            document["top_k"] = self.top_k
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "StrategyPortfolioSpec":
        top_k = document.get("top_k")
        portfolio_construction = PortfolioConstructionSpec.from_document(
            document.get("portfolio_construction")
        )
        rebalance_interval_steps = _rebalance_interval_steps_from_document(document)
        return cls(
            portfolio_construction=portfolio_construction,
            rebalance_friction_policy=RebalanceFrictionPolicySpec.from_document(
                dict(document.get("rebalance_friction_policy", {}))
            ),
            execution_policy=ExecutionPolicySpec.from_document(
                dict(document.get("execution_policy", {}))
            ),
            holding_cost_policy=HoldingCostPolicySpec.from_document(
                dict(document.get("holding_cost_policy", {}))
            ),
            selection_kind=str(document.get("selection_kind", "all_assets")),
            top_k=None if top_k is None else int(top_k),
            rebalance_interval_steps=int(rebalance_interval_steps),
        )

@dataclass(frozen=True)
class TradingStrategySpec:
    strategy_id: str
    label: str
    scope: TradingStrategyScopeSpec
    signal_discovery_id: str | None
    position_rule_id: str
    family_mix: str | None
    portfolio: StrategyPortfolioSpec
    created_at: str
    adaptation_policy: AdaptationPolicySpec = field(
        default_factory=lambda: AdaptationPolicySpec(
            enabled=False,
            adaptation_blend=0.2,
        )
    )

    def to_document(self) -> dict[str, Any]:
        document = {
            "strategy_id": self.strategy_id,
            "label": self.label,
            "scope": self.scope.to_document(),
            "signal_discovery_id": self.signal_discovery_id,
            "position_rule_id": self.position_rule_id,
            "family_mix": self.family_mix,
            "portfolio": self.portfolio.to_document(),
            "adaptation_policy": self.adaptation_policy.to_document(),
            "created_at": self.created_at,
        }
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "TradingStrategySpec":
        portfolio_document = document.get("portfolio")
        if not isinstance(portfolio_document, dict):
            raise ValueError("trading strategy portfolio is required")
        portfolio = StrategyPortfolioSpec.from_document(dict(portfolio_document))
        return cls(
            strategy_id=str(document["strategy_id"]),
            label=str(document["label"]),
            scope=TradingStrategyScopeSpec.from_document(dict(document.get("scope", {}))),
            signal_discovery_id=_normalize_optional(
                None
                if document.get("signal_discovery_id") is None
                else str(document["signal_discovery_id"])
            ),
            position_rule_id=str(document.get("position_rule_id", "constant_hold")),
            family_mix=_normalize_optional(
                None if document.get("family_mix") is None else str(document["family_mix"])
            ),
            adaptation_policy=AdaptationPolicySpec.from_document(
                dict(document.get("adaptation_policy", {}))
            ),
            portfolio=portfolio,
            created_at=str(document["created_at"]),
        )

    @property
    def subject_set_id(self) -> str | None:
        return self.scope.subject_set_id

    @property
    def target_id(self) -> str | None:
        return self.scope.target_id

    @property
    def selection_kind(self) -> str:
        return self.portfolio.selection_kind

    @property
    def rebalance_friction_policy(self) -> RebalanceFrictionPolicySpec:
        return self.portfolio.rebalance_friction_policy

    @property
    def execution_policy(self) -> ExecutionPolicySpec:
        return self.portfolio.execution_policy

    @property
    def holding_cost_policy(self) -> HoldingCostPolicySpec:
        return self.portfolio.holding_cost_policy

    @property
    def portfolio_construction(self) -> PortfolioConstructionSpec:
        return self.portfolio.portfolio_construction

    @property
    def sleeve_composition(self) -> StrategySleeveCompositionSpec | None:
        return self.portfolio.portfolio_construction.sleeve_composition
