from __future__ import annotations

from dataclasses import dataclass, field

from .contract_boundaries import (
    PortfolioConstraintBoundary,
    default_portfolio_constraint_boundary,
)
from .portfolio_direction import normalize_portfolio_direction_mode


SIZING_METHODS = (
    "signal_weighted",
    "constrained_signal_weighted",
    "signed_mean_variance",
    "equal_weight",
    "minimum_variance",
    "risk_budgeting",
    "hierarchical_risk_parity",
    "conviction_adjusted_hierarchical_risk_parity",
    "diversified_risk_budget",
)


@dataclass(frozen=True)
class PortfolioConstructionSpec:
    sizing_method: str = "signal_weighted"
    rebalance_interval_steps: int = 1
    long_only: bool = False
    direction_mode: str | None = None
    active_weight_budget: float | None = None
    gross_exposure_cap: float | None = None
    target_vol: float | None = None
    gross_leverage_cap: float | None = None
    net_exposure_target: float | None = None
    asset_class_weight_caps: dict[str, float] = field(default_factory=dict)
    cluster_weight_caps: dict[str, float] = field(default_factory=dict)
    effective_n_floor: float | None = None
    top_gross_share_cap_n: int | None = None
    top_gross_share_cap: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sizing_method, str) or not self.sizing_method:
            raise ValueError(
                "portfolio_construction.sizing_method must be a string"
            )
        if self.sizing_method not in SIZING_METHODS:
            raise ValueError(
                "portfolio_construction.sizing_method must be one of: "
                + ", ".join(SIZING_METHODS)
            )
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
        if self.active_weight_budget is not None:
            if not isinstance(self.active_weight_budget, int | float):
                raise ValueError(
                    "portfolio_construction.active_weight_budget must be numeric"
                )
            if self.active_weight_budget < 0.0 or self.active_weight_budget > 1.0:
                raise ValueError(
                    "portfolio_construction.active_weight_budget must be in [0, 1]"
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
        if self.effective_n_floor is not None:
            if not isinstance(self.effective_n_floor, int | float):
                raise ValueError(
                    "portfolio_construction.effective_n_floor must be numeric"
                )
            if float(self.effective_n_floor) < 0.0:
                raise ValueError(
                    "portfolio_construction.effective_n_floor must be >= 0"
                )
        if self.top_gross_share_cap_n is not None:
            if (
                not isinstance(self.top_gross_share_cap_n, int)
                or self.top_gross_share_cap_n < 1
            ):
                raise ValueError(
                    "portfolio_construction.top_gross_share_cap_n must be >= 1"
                )
        if self.top_gross_share_cap is not None:
            if not isinstance(self.top_gross_share_cap, int | float):
                raise ValueError(
                    "portfolio_construction.top_gross_share_cap must be numeric"
                )
            if not 0.0 <= float(self.top_gross_share_cap) <= 1.0:
                raise ValueError(
                    "portfolio_construction.top_gross_share_cap must be in [0, 1]"
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

    @property
    def constraint_boundary(self) -> PortfolioConstraintBoundary:
        return default_portfolio_constraint_boundary()
