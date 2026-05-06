from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .contract_boundaries import (
    PortfolioConstraintBoundary,
    default_portfolio_constraint_boundary,
)
from .portfolio_construction_config import (
    PortfolioConstructionSizingSpec,
    PortfolioConstructionSpec,
)
from .portfolio_direction import normalize_portfolio_direction_mode
from .strategy_execution import (
    StrategyExecutionKind,
    StrategyExecutionSpec,
    resolve_strategy_execution_spec,
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
class SelectionPolicySpec:
    selection_kind: str
    top_k: int | None

    def to_document(self) -> dict[str, Any]:
        return {
            "selection_kind": self.selection_kind,
            "top_k": self.top_k,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "SelectionPolicySpec":
        top_k = document.get("top_k")
        return cls(
            selection_kind=str(document.get("selection_kind", "all_assets")),
            top_k=None if top_k is None else int(top_k),
        )


@dataclass(frozen=True)
class SizingPolicySpec:
    sizing_method: str | None

    def to_document(self) -> dict[str, Any]:
        return {
            "sizing_method": self.sizing_method,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "SizingPolicySpec":
        return cls(
            sizing_method=_normalize_optional(
                None
                if document.get("sizing_method") is None
                else str(document["sizing_method"])
            ),
        )


@dataclass(frozen=True)
class RebalancePolicySpec:
    rebalance: str | None
    rebalance_interval_steps: int | None = None

    def __post_init__(self) -> None:
        if self.rebalance_interval_steps is not None and (
            not isinstance(self.rebalance_interval_steps, int)
            or self.rebalance_interval_steps < 1
        ):
            raise ValueError("rebalance_policy.rebalance_interval_steps must be >= 1")

    def to_document(self) -> dict[str, Any]:
        document = {
            "rebalance": self.rebalance,
        }
        if self.rebalance_interval_steps is not None:
            document["rebalance_interval_steps"] = self.rebalance_interval_steps
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "RebalancePolicySpec":
        return cls(
            rebalance=_normalize_optional(
                None if document.get("rebalance") is None else str(document["rebalance"])
            ),
            rebalance_interval_steps=(
                None
                if document.get("rebalance_interval_steps") is None
                else int(document.get("rebalance_interval_steps"))
            ),
        )


@dataclass(frozen=True)
class RiskPolicySpec:
    long_only: bool | None
    gross_exposure_cap: float | None
    direction_mode: str | None = None
    target_vol: float | None = None
    gross_leverage_cap: float | None = None
    net_exposure_target: float | None = None
    asset_class_weight_caps: dict[str, float] = field(default_factory=dict)
    cluster_weight_caps: dict[str, float] = field(default_factory=dict)

    @property
    def constraint_boundary(self) -> PortfolioConstraintBoundary:
        return default_portfolio_constraint_boundary()

    def to_document(self) -> dict[str, Any]:
        document = {
            "gross_exposure_cap": self.gross_exposure_cap,
        }
        direction_mode = self.direction_mode
        if direction_mode is None and self.long_only is not None:
            direction_mode = "long_only" if self.long_only else "long_short"
        if direction_mode is not None:
            document["direction_mode"] = direction_mode
        if self.target_vol is not None:
            document["target_vol"] = self.target_vol
        if self.gross_leverage_cap is not None:
            document["gross_leverage_cap"] = self.gross_leverage_cap
        if self.net_exposure_target is not None:
            document["net_exposure_target"] = self.net_exposure_target
        if self.asset_class_weight_caps:
            document["asset_class_weight_caps"] = dict(self.asset_class_weight_caps)
        if self.cluster_weight_caps:
            document["cluster_weight_caps"] = dict(self.cluster_weight_caps)
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "RiskPolicySpec":
        gross_exposure_cap = document.get("gross_exposure_cap")
        target_vol = document.get("target_vol")
        gross_leverage_cap = document.get("gross_leverage_cap")
        net_exposure_target = document.get("net_exposure_target")
        long_only = document.get("long_only")
        if isinstance(long_only, str):
            normalized_long_only = _normalize_optional(long_only)
            if normalized_long_only is None:
                long_only = None
            elif normalized_long_only == "true":
                long_only = True
            elif normalized_long_only == "false":
                long_only = False
            else:
                raise ValueError(f"unsupported long_only value: {long_only}")
        direction_mode = document.get("direction_mode")
        if direction_mode is not None:
            direction_mode = normalize_portfolio_direction_mode(
                str(direction_mode),
                long_only=bool(long_only) if long_only is not None else False,
            )
            long_only = direction_mode == "long_only"
        return cls(
            long_only=None if long_only is None else bool(long_only),
            gross_exposure_cap=(
                None if gross_exposure_cap is None else float(gross_exposure_cap)
            ),
            direction_mode=direction_mode,
            target_vol=None if target_vol is None else float(target_vol),
            gross_leverage_cap=(
                None if gross_leverage_cap is None else float(gross_leverage_cap)
            ),
            net_exposure_target=(
                None if net_exposure_target is None else float(net_exposure_target)
            ),
            asset_class_weight_caps=_normalized_weight_caps(
                document,
                "asset_class_weight_caps",
            ),
            cluster_weight_caps=_normalized_weight_caps(document, "cluster_weight_caps"),
        )


@dataclass(frozen=True)
class PortfolioPolicySpec:
    selection_policy: SelectionPolicySpec
    sizing_policy: SizingPolicySpec
    rebalance_policy: RebalancePolicySpec
    risk_policy: RiskPolicySpec

    def to_document(self) -> dict[str, Any]:
        return {
            "selection_policy": self.selection_policy.to_document(),
            "sizing_policy": self.sizing_policy.to_document(),
            "rebalance_policy": self.rebalance_policy.to_document(),
            "risk_policy": self.risk_policy.to_document(),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "PortfolioPolicySpec":
        return cls(
            selection_policy=SelectionPolicySpec.from_document(
                dict(document.get("selection_policy", {}))
            ),
            sizing_policy=SizingPolicySpec.from_document(
                dict(document.get("sizing_policy", {}))
            ),
            rebalance_policy=RebalancePolicySpec.from_document(
                dict(document.get("rebalance_policy", {}))
            ),
            risk_policy=RiskPolicySpec.from_document(
                dict(document.get("risk_policy", {}))
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


def _rebalance_interval_steps_from_policy(policy: RebalancePolicySpec) -> int:
    if policy.rebalance_interval_steps is not None:
        return policy.rebalance_interval_steps
    rebalance = policy.rebalance
    if rebalance in {None, "", "-", "none"}:
        return 1
    prefix = "every_"
    suffix = "_steps"
    if rebalance.startswith(prefix) and rebalance.endswith(suffix):
        return int(rebalance[len(prefix) : -len(suffix)])
    return 1


def _rebalance_policy_from_interval(rebalance_interval_steps: int) -> RebalancePolicySpec:
    return RebalancePolicySpec(
        rebalance=f"every_{int(rebalance_interval_steps)}_steps",
        rebalance_interval_steps=int(rebalance_interval_steps),
    )


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

    def __post_init__(self) -> None:
        top_k = self.top_k
        construction_top_k = getattr(self.portfolio_construction, "top_k", None)
        if top_k is None and construction_top_k is not None:
            top_k = int(construction_top_k)
        if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
            raise ValueError("strategy portfolio top_k must be >= 1")
        object.__setattr__(self, "top_k", top_k)

    def to_document(self) -> dict[str, Any]:
        document = {
            "portfolio_construction": self.portfolio_construction.to_document(),
            "rebalance_friction_policy": self.rebalance_friction_policy.to_document(),
            "execution_policy": self.execution_policy.to_document(),
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
        return cls(
            portfolio_construction=PortfolioConstructionSpec.from_document(
                document.get("portfolio_construction")
            ),
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
        )

    @classmethod
    def from_legacy(
        cls,
        *,
        portfolio_policy: PortfolioPolicySpec,
        rebalance_friction_policy: RebalanceFrictionPolicySpec,
        execution_policy: ExecutionPolicySpec,
        holding_cost_policy: HoldingCostPolicySpec,
        portfolio_construction: PortfolioConstructionSpec | None,
        sleeve_composition: StrategySleeveCompositionSpec | None,
    ) -> "StrategyPortfolioSpec":
        if portfolio_construction is not None:
            construction = portfolio_construction
        else:
            risk_policy = portfolio_policy.risk_policy
            sizing_method = portfolio_policy.sizing_policy.sizing_method or "equal_weight"
            construction = PortfolioConstructionSpec(
                sizing_policy=PortfolioConstructionSizingSpec(
                    sizing_method=sizing_method,
                ),
                rebalance_interval_steps=_rebalance_interval_steps_from_policy(
                    portfolio_policy.rebalance_policy
                ),
                long_only=(
                    False
                    if risk_policy.long_only is None
                    else risk_policy.long_only
                ),
                direction_mode=risk_policy.direction_mode,
                gross_exposure_cap=risk_policy.gross_exposure_cap,
                target_vol=risk_policy.target_vol,
                gross_leverage_cap=risk_policy.gross_leverage_cap,
                net_exposure_target=risk_policy.net_exposure_target,
                asset_class_weight_caps=dict(risk_policy.asset_class_weight_caps),
                cluster_weight_caps=dict(risk_policy.cluster_weight_caps),
                sleeve_composition=sleeve_composition,
            )
        return cls(
            portfolio_construction=construction,
            rebalance_friction_policy=rebalance_friction_policy,
            execution_policy=execution_policy,
            holding_cost_policy=holding_cost_policy,
            selection_kind=portfolio_policy.selection_policy.selection_kind,
            top_k=portfolio_policy.selection_policy.top_k,
        )

    def to_portfolio_policy(self) -> PortfolioPolicySpec:
        construction = self.portfolio_construction
        return PortfolioPolicySpec(
            selection_policy=SelectionPolicySpec(
                selection_kind=self.selection_kind,
                top_k=self.top_k,
            ),
            sizing_policy=SizingPolicySpec(
                sizing_method=construction.sizing_method,
            ),
            rebalance_policy=_rebalance_policy_from_interval(
                construction.rebalance_interval_steps
            ),
            risk_policy=RiskPolicySpec(
                long_only=construction.long_only,
                gross_exposure_cap=construction.gross_exposure_cap,
                direction_mode=construction.direction_mode,
                target_vol=construction.target_vol,
                gross_leverage_cap=construction.gross_leverage_cap,
                net_exposure_target=construction.net_exposure_target,
                asset_class_weight_caps=dict(construction.asset_class_weight_caps),
                cluster_weight_caps=dict(construction.cluster_weight_caps),
            ),
        )


@dataclass(frozen=True)
class TradingStrategySpec:
    strategy_id: str
    label: str
    scope: TradingStrategyScopeSpec
    signal_discovery_id: str | None
    position_rule_id: str
    family_mix: str | None
    execution_kind: StrategyExecutionKind
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
            "execution_kind": self.execution_kind,
            "portfolio": self.portfolio.to_document(),
            "adaptation_policy": self.adaptation_policy.to_document(),
            "created_at": self.created_at,
        }
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "TradingStrategySpec":
        portfolio_document = document.get("portfolio")
        portfolio = (
            StrategyPortfolioSpec.from_document(dict(portfolio_document))
            if isinstance(portfolio_document, dict)
            else None
        )
        legacy_portfolio_construction = (
            PortfolioConstructionSpec.from_document(dict(document["portfolio_construction"]))
            if document.get("portfolio_construction") is not None
            else None
        )
        legacy_portfolio_policy = PortfolioPolicySpec.from_document(
            dict(document.get("portfolio_policy", {}))
        )
        legacy_rebalance_friction_policy = RebalanceFrictionPolicySpec.from_document(
            dict(document.get("rebalance_friction_policy", {}))
        )
        legacy_execution_policy = ExecutionPolicySpec.from_document(
            dict(document.get("execution_policy", {}))
        )
        legacy_holding_cost_policy = HoldingCostPolicySpec.from_document(
            dict(document.get("holding_cost_policy", {}))
        )
        legacy_sleeve_composition = StrategySleeveCompositionSpec.from_document(
            None
            if document.get("sleeve_composition") is None
            else dict(document["sleeve_composition"])
        )
        resolved_portfolio = portfolio or StrategyPortfolioSpec.from_legacy(
            portfolio_policy=legacy_portfolio_policy,
            rebalance_friction_policy=legacy_rebalance_friction_policy,
            execution_policy=legacy_execution_policy,
            holding_cost_policy=legacy_holding_cost_policy,
            portfolio_construction=legacy_portfolio_construction,
            sleeve_composition=legacy_sleeve_composition,
        )
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
            execution_kind=str(document.get("execution_kind", "trainless")),
            adaptation_policy=AdaptationPolicySpec.from_document(
                dict(document.get("adaptation_policy", {}))
            ),
            portfolio=resolved_portfolio,
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
        return self.portfolio_policy.selection_policy.selection_kind

    @property
    def portfolio_policy(self) -> PortfolioPolicySpec:
        return self.portfolio.to_portfolio_policy()

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

    @property
    def execution(self) -> StrategyExecutionSpec:
        return resolve_strategy_execution_spec(
            {
                "signal_discovery": self.signal_discovery_id or "",
                "execution_kind": self.execution_kind,
                "position_rule": self.position_rule_id,
            }
        )

    @property
    def requires_signal_train(self) -> bool:
        return self.execution.requires_signal_train
