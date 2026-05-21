from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contract_boundaries import (
    PortfolioConstraintBoundary,
    default_portfolio_constraint_boundary,
)
from .portfolio_direction import normalize_portfolio_direction_mode
from .portfolio_overlay import ActiveOverlaySpec
from .strategy_sleeves import StrategySleeveCompositionSpec


SIZING_METHODS = (
    "signal_weighted",
    "signed_mean_variance",
    "equal_weight",
    "minimum_variance",
    "risk_budgeting",
    "hierarchical_risk_parity",
    "conviction_adjusted_hierarchical_risk_parity",
    "diversified_risk_budget",
)


SIZING_ENGINES = (
    "rule_based",
    "optimizer",
    "history_based",
)


SIZING_FAMILIES = (
    "signed_optimizer",
    "risk_budget_allocator",
)


RISK_NORMALIZATION_MODES = (
    "none",
    "gross",
)


CONSTRUCTION_KINDS = (
    "active_portfolio",
    "hold_baseline",
)


def inferred_sizing_family(sizing_method: str) -> str:
    if sizing_method in {
        "equal_weight",
        "minimum_variance",
        "risk_budgeting",
        "hierarchical_risk_parity",
        "conviction_adjusted_hierarchical_risk_parity",
        "diversified_risk_budget",
    }:
        return "risk_budget_allocator"
    return "signed_optimizer"


def normalize_construction_kind(value: str | None) -> str:
    construction_kind = "active_portfolio" if value in {None, "", "-"} else str(value)
    if construction_kind not in CONSTRUCTION_KINDS:
        raise ValueError(
            "portfolio_construction.construction_kind must be one of: "
            + ", ".join(CONSTRUCTION_KINDS)
        )
    return construction_kind


def _normalized_weight_caps(
    document: dict[str, Any],
    field_name: str,
) -> dict[str, float]:
    raw_value = document.get(field_name)
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError(f"portfolio_construction.{field_name} must be an object")
    normalized: dict[str, float] = {}
    for group_name, cap_value in raw_value.items():
        if not isinstance(group_name, str) or not group_name:
            raise ValueError(
                f"portfolio_construction.{field_name} keys must be non-empty strings"
            )
        if not isinstance(cap_value, int | float):
            raise ValueError(
                f"portfolio_construction.{field_name}[{group_name}] must be numeric"
            )
        if float(cap_value) < 0.0:
            raise ValueError(
                f"portfolio_construction.{field_name}[{group_name}] must be >= 0"
            )
        normalized[str(group_name)] = float(cap_value)
    return normalized


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
    raise ValueError(f"portfolio_construction.risk_budget.{field_name} must be boolean")


@dataclass(frozen=True)
class PortfolioConstructionSizingSpec:
    sizing_method: str = "signal_weighted"
    sizing_engine: str | None = None
    sizing_family: str | None = None

    def __post_init__(self) -> None:
        sizing_method = self.sizing_method
        sizing_engine = self.sizing_engine
        sizing_family = self.sizing_family
        if sizing_engine is None:
            sizing_engine = (
                "rule_based"
                if sizing_method == "signal_weighted"
                else "optimizer"
                if sizing_method == "signed_mean_variance"
                else "history_based"
            )
            object.__setattr__(self, "sizing_engine", sizing_engine)
        if sizing_family is None:
            sizing_family = inferred_sizing_family(sizing_method)
            object.__setattr__(self, "sizing_family", sizing_family)
        if not isinstance(sizing_method, str) or not sizing_method:
            raise ValueError(
                "portfolio_construction.sizing_policy.sizing_method "
                "must be a string"
            )
        if not isinstance(sizing_engine, str) or not sizing_engine:
            raise ValueError(
                "portfolio_construction.sizing_policy.sizing_engine "
                "must be a string"
            )
        if sizing_method not in SIZING_METHODS:
            raise ValueError(
                "portfolio_construction.sizing_policy.sizing_method "
                "must be one of: "
                + ", ".join(SIZING_METHODS)
            )
        if sizing_engine not in SIZING_ENGINES:
            raise ValueError(
                "portfolio_construction.sizing_policy.sizing_engine "
                "must be one of: "
                + ", ".join(SIZING_ENGINES)
            )
        if sizing_family not in SIZING_FAMILIES:
            raise ValueError(
                "portfolio_construction.sizing_policy.sizing_family "
                "must be one of: "
                + ", ".join(SIZING_FAMILIES)
            )
        expected_sizing_family = inferred_sizing_family(sizing_method)
        if sizing_family != expected_sizing_family:
            raise ValueError(
                "portfolio_construction.sizing_policy.sizing_family must match "
                f"{sizing_method}: {expected_sizing_family}"
            )
        if sizing_method == "signal_weighted":
            if sizing_engine not in {"rule_based", "optimizer"}:
                raise ValueError(
                    "signal_weighted sizing_method requires "
                    "rule_based or optimizer sizing_engine"
                )
        elif sizing_method == "signed_mean_variance":
            if sizing_engine != "optimizer":
                raise ValueError(
                    "signed_mean_variance sizing_method requires "
                    "sizing_engine=optimizer"
                )
        elif sizing_engine != "history_based":
            raise ValueError(
                "history-based sizing methods require sizing_engine=history_based"
            )

    def to_document(self) -> dict[str, Any]:
        return {
            "sizing_method": self.sizing_method,
            "sizing_engine": self.sizing_engine,
            "sizing_family": self.sizing_family,
        }

    @classmethod
    def from_document(
        cls,
        document: dict[str, Any] | None,
    ) -> "PortfolioConstructionSizingSpec":
        if document is None:
            return cls()
        if not isinstance(document, dict):
            raise ValueError("portfolio_construction.sizing_policy must be an object")
        return cls(
            sizing_method=str(document.get("sizing_method", "signal_weighted")),
            sizing_engine=(
                None
                if document.get("sizing_engine") is None
                else str(document.get("sizing_engine"))
            ),
            sizing_family=(
                None
                if document.get("sizing_family") is None
                else str(document.get("sizing_family"))
            ),
        )


@dataclass(frozen=True)
class PortfolioIntentSpec:
    effective_n_floor: float | None = None
    top_gross_share_cap_n: int | None = None
    top_gross_share_cap: float | None = None
    concentration_min_abs_weight: float = 0.001

    def __post_init__(self) -> None:
        if self.effective_n_floor is not None:
            if not isinstance(self.effective_n_floor, int | float):
                raise ValueError(
                    "portfolio_construction.portfolio_intent.effective_n_floor "
                    "must be numeric"
                )
            if float(self.effective_n_floor) < 0.0:
                raise ValueError(
                    "portfolio_construction.portfolio_intent.effective_n_floor "
                    "must be >= 0"
                )
        if self.top_gross_share_cap_n is not None:
            if (
                not isinstance(self.top_gross_share_cap_n, int)
                or self.top_gross_share_cap_n < 1
            ):
                raise ValueError(
                    "portfolio_construction.portfolio_intent.top_gross_share_cap_n "
                    "must be >= 1"
                )
        if self.top_gross_share_cap is not None:
            if not isinstance(self.top_gross_share_cap, int | float):
                raise ValueError(
                    "portfolio_construction.portfolio_intent.top_gross_share_cap "
                    "must be numeric"
                )
            if not 0.0 <= float(self.top_gross_share_cap) <= 1.0:
                raise ValueError(
                    "portfolio_construction.portfolio_intent.top_gross_share_cap "
                    "must be in [0, 1]"
                )
        if not isinstance(self.concentration_min_abs_weight, int | float):
            raise ValueError(
                "portfolio_construction.portfolio_intent.concentration_min_abs_weight "
                "must be numeric"
            )
        if float(self.concentration_min_abs_weight) < 0.0:
            raise ValueError(
                "portfolio_construction.portfolio_intent.concentration_min_abs_weight "
                "must be >= 0"
            )

    @property
    def has_constraints(self) -> bool:
        return (
            self.effective_n_floor is not None
            or (
                self.top_gross_share_cap_n is not None
                and self.top_gross_share_cap is not None
            )
        )

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {}
        if self.effective_n_floor is not None:
            document["effective_n_floor"] = self.effective_n_floor
        if self.top_gross_share_cap_n is not None:
            document["top_gross_share_cap_n"] = self.top_gross_share_cap_n
        if self.top_gross_share_cap is not None:
            document["top_gross_share_cap"] = self.top_gross_share_cap
        if self.concentration_min_abs_weight != 0.001:
            document["concentration_min_abs_weight"] = self.concentration_min_abs_weight
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any] | None) -> "PortfolioIntentSpec":
        if document is None:
            return cls()
        if not isinstance(document, dict):
            raise ValueError("portfolio_construction.portfolio_intent must be an object")
        return cls(
            effective_n_floor=(
                None
                if document.get("effective_n_floor") is None
                else float(document.get("effective_n_floor"))
            ),
            top_gross_share_cap_n=(
                None
                if document.get("top_gross_share_cap_n") is None
                else int(document.get("top_gross_share_cap_n"))
            ),
            top_gross_share_cap=(
                None
                if document.get("top_gross_share_cap") is None
                else float(document.get("top_gross_share_cap"))
            ),
            concentration_min_abs_weight=float(
                document.get("concentration_min_abs_weight", 0.001)
            ),
        )


@dataclass(frozen=True)
class PortfolioRiskBudgetSpec:
    risk_normalization_mode: str = "none"
    target_gross_exposure: float | None = None
    allow_releverage: bool = False

    def __post_init__(self) -> None:
        mode = str(self.risk_normalization_mode)
        if mode not in RISK_NORMALIZATION_MODES:
            raise ValueError(
                "portfolio_construction.risk_budget.risk_normalization_mode "
                "must be one of: "
                + ", ".join(RISK_NORMALIZATION_MODES)
            )
        object.__setattr__(self, "risk_normalization_mode", mode)
        if self.target_gross_exposure is not None:
            if not isinstance(self.target_gross_exposure, int | float):
                raise ValueError(
                    "portfolio_construction.risk_budget.target_gross_exposure "
                    "must be numeric"
                )
            if float(self.target_gross_exposure) < 0.0:
                raise ValueError(
                    "portfolio_construction.risk_budget.target_gross_exposure "
                    "must be >= 0"
                )
            object.__setattr__(
                self,
                "target_gross_exposure",
                float(self.target_gross_exposure),
            )
        if not isinstance(self.allow_releverage, bool):
            raise ValueError(
                "portfolio_construction.risk_budget.allow_releverage must be boolean"
            )

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {}
        if self.risk_normalization_mode != "none":
            document["risk_normalization_mode"] = self.risk_normalization_mode
        if self.target_gross_exposure is not None:
            document["target_gross_exposure"] = self.target_gross_exposure
        if self.allow_releverage:
            document["allow_releverage"] = self.allow_releverage
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any] | None) -> "PortfolioRiskBudgetSpec":
        if document is None:
            return cls()
        if not isinstance(document, dict):
            raise ValueError("portfolio_construction.risk_budget must be an object")
        return cls(
            risk_normalization_mode=str(
                document.get("risk_normalization_mode", "none")
            ),
            target_gross_exposure=(
                None
                if document.get("target_gross_exposure") is None
                else float(document.get("target_gross_exposure"))
            ),
            allow_releverage=_optional_bool_from_document(
                document,
                "allow_releverage",
                default=False,
            ),
        )


@dataclass(frozen=True)
class PortfolioConstructionSpec:
    construction_kind: str = "active_portfolio"
    sizing_policy: PortfolioConstructionSizingSpec = field(
        default_factory=PortfolioConstructionSizingSpec
    )
    rebalance_interval_steps: int = 1
    long_only: bool = False
    direction_mode: str | None = None
    active_overlay: ActiveOverlaySpec | None = field(default_factory=ActiveOverlaySpec)
    gross_exposure_cap: float | None = None
    target_vol: float | None = None
    gross_leverage_cap: float | None = None
    net_exposure_target: float | None = None
    asset_class_weight_caps: dict[str, float] = field(default_factory=dict)
    cluster_weight_caps: dict[str, float] = field(default_factory=dict)
    portfolio_intent: PortfolioIntentSpec = field(default_factory=PortfolioIntentSpec)
    risk_budget: PortfolioRiskBudgetSpec = field(default_factory=PortfolioRiskBudgetSpec)
    sleeve_composition: StrategySleeveCompositionSpec | None = None

    def __post_init__(self) -> None:
        construction_kind = normalize_construction_kind(self.construction_kind)
        object.__setattr__(self, "construction_kind", construction_kind)
        if (
            not isinstance(self.rebalance_interval_steps, int)
            or self.rebalance_interval_steps < 1
        ):
            raise ValueError(
                "portfolio_construction.rebalance_interval_steps must be >= 1"
            )
        if not isinstance(self.long_only, bool):
            raise ValueError("portfolio_construction.long_only must be boolean")
        direction_mode = normalize_portfolio_direction_mode(
            self.direction_mode,
            long_only=self.long_only,
        )
        object.__setattr__(self, "direction_mode", direction_mode)
        object.__setattr__(self, "long_only", direction_mode == "long_only")
        if self.active_overlay is not None and not isinstance(
            self.active_overlay,
            ActiveOverlaySpec,
        ):
            raise ValueError(
                "portfolio_construction.active_overlay must be ActiveOverlaySpec"
            )
        if self.gross_exposure_cap is not None and not isinstance(
            self.gross_exposure_cap, int | float
        ):
            raise ValueError(
                "portfolio_construction.gross_exposure_cap must be numeric"
            )
        if self.target_vol is not None and not isinstance(self.target_vol, int | float):
            raise ValueError("portfolio_construction.target_vol must be numeric")
        if self.gross_leverage_cap is not None and not isinstance(
            self.gross_leverage_cap,
            int | float,
        ):
            raise ValueError("portfolio_construction.gross_leverage_cap must be numeric")
        if self.net_exposure_target is not None and not isinstance(
            self.net_exposure_target,
            int | float,
        ):
            raise ValueError("portfolio_construction.net_exposure_target must be numeric")
        if not isinstance(self.portfolio_intent, PortfolioIntentSpec):
            object.__setattr__(
                self,
                "portfolio_intent",
                PortfolioIntentSpec.from_document(
                    _document_from_compatible_spec(self.portfolio_intent)
                ),
            )
        if not isinstance(self.portfolio_intent, PortfolioIntentSpec):
            raise ValueError(
                "portfolio_construction.portfolio_intent must be PortfolioIntentSpec"
            )
        if not isinstance(self.risk_budget, PortfolioRiskBudgetSpec):
            object.__setattr__(
                self,
                "risk_budget",
                PortfolioRiskBudgetSpec.from_document(
                    _document_from_compatible_spec(self.risk_budget)
                ),
            )
        if not isinstance(self.risk_budget, PortfolioRiskBudgetSpec):
            raise ValueError(
                "portfolio_construction.risk_budget must be PortfolioRiskBudgetSpec"
            )
        for field_name, caps in (
            ("asset_class_weight_caps", self.asset_class_weight_caps),
            ("cluster_weight_caps", self.cluster_weight_caps),
        ):
            if not isinstance(caps, dict):
                raise ValueError(f"portfolio_construction.{field_name} must be an object")
            for group_name, cap_value in caps.items():
                if not isinstance(group_name, str) or not group_name:
                    raise ValueError(
                        f"portfolio_construction.{field_name} keys must be non-empty strings"
                    )
                if not isinstance(cap_value, int | float):
                    raise ValueError(
                        f"portfolio_construction.{field_name}[{group_name}] must be numeric"
                    )
                if float(cap_value) < 0.0:
                    raise ValueError(
                        f"portfolio_construction.{field_name}[{group_name}] must be >= 0"
                    )
        if construction_kind == "hold_baseline":
            if self.active_overlay is not None:
                raise ValueError(
                    "hold_baseline portfolio_construction must not define active_overlay"
                )
            if self.target_vol is not None:
                raise ValueError(
                    "hold_baseline portfolio_construction must not define target_vol"
                )
            if self.asset_class_weight_caps:
                raise ValueError(
                    "hold_baseline portfolio_construction must not define asset_class_weight_caps"
                )
            if self.cluster_weight_caps:
                raise ValueError(
                    "hold_baseline portfolio_construction must not define cluster_weight_caps"
                )
            if self.portfolio_intent.has_constraints:
                raise ValueError(
                    "hold_baseline portfolio_construction must not define portfolio_intent constraints"
                )
            if self.risk_budget.target_gross_exposure is not None:
                raise ValueError(
                    "hold_baseline portfolio_construction must not define target_gross_exposure"
                )
            if self.risk_budget.allow_releverage:
                raise ValueError(
                    "hold_baseline portfolio_construction must not allow releverage"
                )
            if self.sleeve_composition is not None:
                raise ValueError(
                    "hold_baseline portfolio_construction must not define sleeve_composition"
                )

    @property
    def sizing_method(self) -> str:
        return self.sizing_policy.sizing_method

    @property
    def sizing_engine(self) -> str:
        return self.sizing_policy.sizing_engine

    @property
    def constraint_boundary(self) -> PortfolioConstraintBoundary:
        return default_portfolio_constraint_boundary()

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "construction_kind": self.construction_kind,
            "sizing_policy": self.sizing_policy.to_document(),
            "direction_mode": self.direction_mode,
        }
        if self.active_overlay is not None:
            document["active_overlay"] = self.active_overlay.to_document()
        if self.gross_exposure_cap is not None:
            document["gross_exposure_cap"] = self.gross_exposure_cap
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
        if self.portfolio_intent.to_document():
            document["portfolio_intent"] = self.portfolio_intent.to_document()
        if self.risk_budget.to_document():
            document["risk_budget"] = self.risk_budget.to_document()
        if self.sleeve_composition is not None:
            document["sleeve_composition"] = self.sleeve_composition.to_document()
        return document

    @classmethod
    def from_document(
        cls, document: dict[str, Any] | None
    ) -> "PortfolioConstructionSpec":
        if document is None:
            return cls()
        if not isinstance(document, dict):
            raise ValueError("portfolio_construction must be an object")
        construction_kind = normalize_construction_kind(
            None
            if document.get("construction_kind") is None
            else str(document.get("construction_kind"))
        )
        return cls(
            construction_kind=construction_kind,
            sizing_policy=PortfolioConstructionSizingSpec.from_document(
                document.get("sizing_policy")
            ),
            long_only=False,
            direction_mode=(
                None
                if document.get("direction_mode") is None
                else str(document.get("direction_mode"))
            ),
            active_overlay=(
                None
                if construction_kind == "hold_baseline"
                and document.get("active_overlay") is None
                else ActiveOverlaySpec.from_document(document.get("active_overlay"))
            ),
            gross_exposure_cap=(
                None
                if document.get("gross_exposure_cap") is None
                else float(document.get("gross_exposure_cap"))
            ),
            target_vol=(
                None
                if document.get("target_vol") is None
                else float(document.get("target_vol"))
            ),
            gross_leverage_cap=(
                None
                if document.get("gross_leverage_cap") is None
                else float(document.get("gross_leverage_cap"))
            ),
            net_exposure_target=(
                None
                if document.get("net_exposure_target") is None
                else float(document.get("net_exposure_target"))
            ),
            asset_class_weight_caps=_normalized_weight_caps(
                document,
                "asset_class_weight_caps",
            ),
            cluster_weight_caps=_normalized_weight_caps(
                document,
                "cluster_weight_caps",
            ),
            portfolio_intent=PortfolioIntentSpec.from_document(
                document.get("portfolio_intent")
            ),
            risk_budget=PortfolioRiskBudgetSpec.from_document(
                document.get("risk_budget")
            ),
            sleeve_composition=StrategySleeveCompositionSpec.from_document(
                (
                    None
                    if document.get("sleeve_composition") is None
                    else dict(document.get("sleeve_composition"))
                )
            ),
        )


def _document_from_compatible_spec(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    to_document = getattr(value, "to_document", None)
    if callable(to_document):
        document = to_document()
        if isinstance(document, dict):
            return document
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return None
