from __future__ import annotations

def _format_constraint_caps(values: dict[str, float]) -> str:
    if not values:
        return "-"
    return ";".join(f"{key}={value}" for key, value in sorted(values.items()))


def _format_sleeve_composition(composition) -> str | None:
    if composition is None:
        return None
    parts = [
        f"{item.sleeve_id}:{item.sleeve_kind}:{item.risk_budget}"
        for item in composition.sleeves
    ]
    return ";".join(parts) if parts else None


def _optimizer_backend_for_sizing(
    sizing_method: str | None,
    sizing_engine: str | None,
) -> str:
    if sizing_method == "signed_mean_variance" and sizing_engine == "optimizer":
        return "cvxpy_signed_mean_variance"
    if sizing_method == "signal_weighted" and sizing_engine == "optimizer":
        return "cvxpy_constrained_optimizer"
    if sizing_engine == "history_based":
        if sizing_method == "equal_weight":
            return "history_based_equal_weight"
        return f"skfolio:{sizing_method or '-'}"
    return "rule_based_signal_weighted"


def build_evaluation_task_contract_fields(
    portfolio_construction,
    *,
    rebalance_friction_policy,
    execution_cost_assumptions,
    holding_cost_assumptions,
    target_id: str | None = None,
    selection_kind: str | None = None,
    top_k: int | None = None,
) -> dict[str, str | int | float | bool]:
    rebalance_interval_steps = portfolio_construction.rebalance_interval_steps
    is_hold_baseline = portfolio_construction.construction_kind == "hold_baseline"
    resolved_top_k = top_k
    resolved_selection = (
        selection_kind
        if selection_kind is not None
        else "all_assets"
        if resolved_top_k is None
        else "top_k"
    )
    fields: dict[str, str | int | float | bool] = {
        "construction_kind": portfolio_construction.construction_kind,
        "target_id": "-" if target_id is None else target_id,
        "selection": resolved_selection,
        "top_k": "-" if resolved_top_k is None else resolved_top_k,
        "sizing": portfolio_construction.sizing_method or "-",
        "optimizer_backend": _optimizer_backend_for_sizing(
            portfolio_construction.sizing_method,
            portfolio_construction.sizing_engine,
        ),
        "rebalance": f"every_{rebalance_interval_steps}_steps",
        "direction_mode": portfolio_construction.direction_mode,
        "gross_exposure_cap": (
            "-"
            if portfolio_construction.gross_exposure_cap is None
            else portfolio_construction.gross_exposure_cap
        ),
        "gross_leverage_cap": (
            "-"
            if portfolio_construction.gross_leverage_cap is None
            else portfolio_construction.gross_leverage_cap
        ),
        "net_exposure_target": (
            "-"
            if portfolio_construction.net_exposure_target is None
            else portfolio_construction.net_exposure_target
        ),
        "turnover_friction": (
            "-"
            if rebalance_friction_policy.turnover_friction is None
            else rebalance_friction_policy.turnover_friction
        ),
        "no_trade_band": (
            "-"
            if rebalance_friction_policy.no_trade_band is None
            else rebalance_friction_policy.no_trade_band
        ),
        "execution_mode": getattr(
            rebalance_friction_policy,
            "execution_mode",
            "utility_priority",
        ),
        "turnover_budget": (
            "-"
            if getattr(rebalance_friction_policy, "turnover_budget", None) is None
            else rebalance_friction_policy.turnover_budget
        ),
        "benefit_scale": getattr(rebalance_friction_policy, "benefit_scale", 1.0),
        "min_trade_utility": getattr(
            rebalance_friction_policy,
            "min_trade_utility",
            0.0,
        ),
        "uncertainty_aversion": getattr(
            rebalance_friction_policy,
            "uncertainty_aversion",
            1.0,
        ),
        "risk_aversion": getattr(rebalance_friction_policy, "risk_aversion", 0.0),
        "partial_fill_enabled": str(
            getattr(rebalance_friction_policy, "partial_fill_enabled", True)
        ).lower(),
        "market_impact_bps": (
            "-"
            if execution_cost_assumptions.market_impact_bps is None
            else execution_cost_assumptions.market_impact_bps
        ),
        "fee_bps": (
            "-"
            if execution_cost_assumptions.fee_bps is None
            else execution_cost_assumptions.fee_bps
        ),
        "funding_bps_per_step": (
            "-"
            if holding_cost_assumptions.funding_bps_per_step is None
            else holding_cost_assumptions.funding_bps_per_step
        ),
        "borrow_fee_bps_per_step": (
            "-"
            if holding_cost_assumptions.borrow_fee_bps_per_step is None
            else holding_cost_assumptions.borrow_fee_bps_per_step
        ),
    }
    if is_hold_baseline:
        if (
            fields["selection"] == "all_assets"
            and fields["sizing"] == "equal_weight"
            and fields["direction_mode"] == "long_only"
        ):
            fields["holding_style"] = "equal_weight_hold"
    else:
        fields["active_overlay"] = (
            "-"
            if portfolio_construction.active_overlay is None
            else portfolio_construction.active_overlay.kind
        )
        fields["active_weight_budget"] = (
            "-"
            if portfolio_construction.active_overlay is None
            else portfolio_construction.active_overlay.active_weight_budget
        )
        fields["sizing_family"] = portfolio_construction.sizing_policy.sizing_family or "-"
        fields["target_vol"] = (
            "-"
            if portfolio_construction.target_vol is None
            else portfolio_construction.target_vol
        )
        fields["risk_normalization_mode"] = (
            portfolio_construction.risk_budget.risk_normalization_mode
        )
        fields["target_gross_exposure"] = (
            "-"
            if portfolio_construction.risk_budget.target_gross_exposure is None
            else portfolio_construction.risk_budget.target_gross_exposure
        )
        fields["allow_releverage"] = str(
            portfolio_construction.risk_budget.allow_releverage
        ).lower()
    if not is_hold_baseline and portfolio_construction.asset_class_weight_caps:
        fields["asset_class_weight_caps"] = _format_constraint_caps(
            portfolio_construction.asset_class_weight_caps
        )
    if not is_hold_baseline and portfolio_construction.cluster_weight_caps:
        fields["cluster_weight_caps"] = _format_constraint_caps(
            portfolio_construction.cluster_weight_caps
        )
    if (
        not is_hold_baseline
        and portfolio_construction.portfolio_intent.effective_n_floor is not None
    ):
        fields["effective_n_floor"] = (
            portfolio_construction.portfolio_intent.effective_n_floor
        )
    if (
        not is_hold_baseline
        and portfolio_construction.portfolio_intent.top_gross_share_cap is not None
    ):
        fields["top_gross_share_cap"] = (
            portfolio_construction.portfolio_intent.top_gross_share_cap
        )
    if (
        not is_hold_baseline
        and portfolio_construction.portfolio_intent.top_gross_share_cap_n is not None
    ):
        fields["top_gross_share_cap_n"] = (
            portfolio_construction.portfolio_intent.top_gross_share_cap_n
        )
    if not is_hold_baseline and portfolio_construction.sleeve_composition is not None:
        fields["sleeve_count"] = len(
            portfolio_construction.sleeve_composition.enabled_sleeves
        )
        sleeves = _format_sleeve_composition(portfolio_construction.sleeve_composition)
        if sleeves is not None:
            fields["sleeves"] = sleeves
    return fields
