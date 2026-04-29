from __future__ import annotations

from typing import Protocol

from .contract_boundaries import active_constraint_stages
from .portfolio_decision import SubjectSet
from .portfolio_construction_config import inferred_sizing_family
from .subject_set_facts import format_subject_set_facts
from .trading_strategy import TradingStrategySpec
from .universe_contract import validate_subject_set_universe_contract


class ReportTradingStrategyState(Protocol):
    @property
    def trading_strategy(self) -> TradingStrategySpec: ...


class ReportSubjectSetState(Protocol):
    @property
    def definition(self) -> SubjectSet: ...


class EvaluationReportStrategyContextReadPort(Protocol):
    def get_trading_strategy(
        self,
        strategy_id: str,
    ) -> ReportTradingStrategyState | None: ...

    def get_subject_set(
        self,
        subject_set_id: str,
    ) -> ReportSubjectSetState | None: ...


def _format_constraint_caps(values: dict[str, float]) -> str:
    if not values:
        return "-"
    return ";".join(f"{key}={value}" for key, value in sorted(values.items()))


def _format_universe_policy_fields(
    universe_policy_fields: dict[str, str | None] | None,
) -> str | None:
    if not universe_policy_fields:
        return None
    parts = [
        f"{key}={value}"
        for key, value in universe_policy_fields.items()
        if value is not None
    ]
    if not parts:
        return None
    return " ".join(parts)


def _format_sleeve_composition(composition) -> str | None:
    if composition is None:
        return None
    parts = [
        f"{item.sleeve_id}:{item.sleeve_kind}:{item.risk_budget}"
        for item in composition.sleeves
    ]
    return ";".join(parts) if parts else None


def _format_sleeves(trading_strategy) -> str | None:
    return _format_sleeve_composition(
        getattr(trading_strategy, "sleeve_composition", None)
    )


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


def _add_subject_set_fields(
    fields: dict[str, str | int | float | bool],
    *,
    subject_set,
    subject_set_id: str | None,
) -> None:
    if subject_set_id is not None:
        fields["subject_set"] = subject_set_id
    if subject_set is None:
        return
    universe_policy = getattr(subject_set, "universe_policy", None)
    if universe_policy is not None:
        if universe_policy.base_currency is not None:
            fields["base_currency"] = universe_policy.base_currency
        if universe_policy.trading_calendar is not None:
            fields["trading_calendar"] = universe_policy.trading_calendar
        if universe_policy.benchmark_id is not None:
            fields["benchmark_id"] = universe_policy.benchmark_id


def build_report_strategy_contract_fields(
    trading_strategy,
    *,
    subject_set=None,
) -> dict[str, str | int | float | bool]:
    portfolio = trading_strategy.portfolio
    portfolio_policy = portfolio.to_portfolio_policy()
    selection_policy = portfolio_policy.selection_policy
    sizing_policy = portfolio_policy.sizing_policy
    rebalance_policy = portfolio_policy.rebalance_policy
    risk_policy = portfolio_policy.risk_policy
    friction_policy = portfolio.rebalance_friction_policy
    execution_policy = portfolio.execution_policy
    holding_cost_policy = portfolio.holding_cost_policy
    fields: dict[str, str | int | float | bool] = {
        "selection": selection_policy.selection_kind,
        "top_k": "-" if selection_policy.top_k is None else selection_policy.top_k,
        "sizing": sizing_policy.sizing_method or "-",
        "sizing_family": (
            "-"
            if sizing_policy.sizing_method is None
            else inferred_sizing_family(sizing_policy.sizing_method)
        ),
        "optimizer_backend": _optimizer_backend_for_sizing(
            sizing_policy.sizing_method,
            getattr(sizing_policy, "sizing_engine", None),
        ),
        "rebalance": rebalance_policy.rebalance or "-",
        "long_only": (
            "-" if risk_policy.long_only is None else str(risk_policy.long_only).lower()
        ),
        "direction_mode": risk_policy.direction_mode or "-",
        "gross_exposure_cap": (
            "-" if risk_policy.gross_exposure_cap is None else risk_policy.gross_exposure_cap
        ),
        "target_vol": "-" if risk_policy.target_vol is None else risk_policy.target_vol,
        "gross_leverage_cap": (
            "-"
            if risk_policy.gross_leverage_cap is None
            else risk_policy.gross_leverage_cap
        ),
        "net_exposure_target": (
            "-"
            if risk_policy.net_exposure_target is None
            else risk_policy.net_exposure_target
        ),
        "turnover_friction": (
            "-"
            if friction_policy.turnover_friction is None
            else friction_policy.turnover_friction
        ),
        "no_trade_band": (
            "-" if friction_policy.no_trade_band is None else friction_policy.no_trade_band
        ),
        "execution_mode": getattr(
            friction_policy,
            "execution_mode",
            "utility_priority",
        ),
        "turnover_budget": (
            "-"
            if getattr(friction_policy, "turnover_budget", None) is None
            else friction_policy.turnover_budget
        ),
        "benefit_scale": getattr(friction_policy, "benefit_scale", 1.0),
        "min_trade_utility": getattr(friction_policy, "min_trade_utility", 0.0),
        "uncertainty_aversion": getattr(friction_policy, "uncertainty_aversion", 1.0),
        "risk_aversion": getattr(friction_policy, "risk_aversion", 0.0),
        "partial_fill_enabled": str(
            getattr(friction_policy, "partial_fill_enabled", True)
        ).lower(),
        "market_impact_bps": (
            "-"
            if execution_policy.market_impact_bps is None
            else execution_policy.market_impact_bps
        ),
        "fee_bps": "-" if execution_policy.fee_bps is None else execution_policy.fee_bps,
        "funding_bps_per_step": (
            "-"
            if holding_cost_policy.funding_bps_per_step is None
            else holding_cost_policy.funding_bps_per_step
        ),
        "borrow_fee_bps_per_step": (
            "-"
            if holding_cost_policy.borrow_fee_bps_per_step is None
            else holding_cost_policy.borrow_fee_bps_per_step
        ),
    }
    _add_subject_set_fields(
        fields,
        subject_set=subject_set,
        subject_set_id=trading_strategy.subject_set_id,
    )
    if portfolio.portfolio_construction.sleeve_composition is not None:
        fields["sleeve_count"] = len(
            portfolio.portfolio_construction.sleeve_composition.enabled_sleeves
        )
        sleeves = _format_sleeves(trading_strategy)
        if sleeves is not None:
            fields["sleeves"] = sleeves
    if risk_policy.asset_class_weight_caps:
        fields["asset_class_weight_caps"] = _format_constraint_caps(
            risk_policy.asset_class_weight_caps
        )
    if risk_policy.cluster_weight_caps:
        fields["cluster_weight_caps"] = _format_constraint_caps(
            risk_policy.cluster_weight_caps
        )
    constraint_stages = active_constraint_stages(
        risk_policy.constraint_boundary,
        field_values={
            "long_only": risk_policy.long_only,
            "gross_exposure_cap": risk_policy.gross_exposure_cap,
            "target_vol": risk_policy.target_vol,
            "gross_leverage_cap": risk_policy.gross_leverage_cap,
            "net_exposure_target": risk_policy.net_exposure_target,
            "asset_class_weight_caps": risk_policy.asset_class_weight_caps,
            "cluster_weight_caps": risk_policy.cluster_weight_caps,
        },
    )
    if constraint_stages:
        fields["constraint_stages"] = ";".join(constraint_stages)
    return fields


def build_report_evaluation_task_contract_fields(
    portfolio_construction,
    *,
    rebalance_friction_policy,
    execution_cost_assumptions,
    holding_cost_assumptions,
    subject_set=None,
    subject_set_id: str | None = None,
) -> dict[str, str | int | float | bool]:
    rebalance_interval_steps = portfolio_construction.rebalance_interval_steps
    is_hold_baseline = portfolio_construction.construction_kind == "hold_baseline"
    fields: dict[str, str | int | float | bool] = {
        "construction_kind": portfolio_construction.construction_kind,
        "selection": "all_assets" if portfolio_construction.top_k is None else "top_k",
        "top_k": (
            "-"
            if portfolio_construction.top_k is None
            else portfolio_construction.top_k
        ),
        "sizing": portfolio_construction.sizing_method or "-",
        "optimizer_backend": _optimizer_backend_for_sizing(
            portfolio_construction.sizing_method,
            portfolio_construction.sizing_engine,
        ),
        "rebalance": f"every_{rebalance_interval_steps}_steps",
        "long_only": str(portfolio_construction.long_only).lower(),
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
            and fields["long_only"] == "true"
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
    _add_subject_set_fields(
        fields,
        subject_set=subject_set,
        subject_set_id=subject_set_id,
    )
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
    constraint_stages = active_constraint_stages(
        portfolio_construction.constraint_boundary,
        field_values={
            "long_only": portfolio_construction.long_only,
            "gross_exposure_cap": portfolio_construction.gross_exposure_cap,
            "target_vol": portfolio_construction.target_vol,
            "gross_leverage_cap": portfolio_construction.gross_leverage_cap,
            "net_exposure_target": portfolio_construction.net_exposure_target,
            "asset_class_weight_caps": portfolio_construction.asset_class_weight_caps,
            "cluster_weight_caps": portfolio_construction.cluster_weight_caps,
        },
    )
    if constraint_stages:
        fields["constraint_stages"] = ";".join(constraint_stages)
    return fields


def format_report_strategy_contract_fields(
    fields: dict[str, str | int | float | bool],
    *,
    subject_set_facts: str | None = None,
    universe_policy_fields: dict[str, str | None] | None = None,
) -> str:
    ordered_keys = (
        "construction_kind",
        "holding_style",
        "selection",
        "top_k",
        "active_overlay",
        "active_weight_budget",
        "sizing",
        "sizing_family",
        "optimizer_backend",
        "rebalance",
        "long_only",
        "direction_mode",
        "gross_exposure_cap",
        "target_vol",
        "gross_leverage_cap",
        "net_exposure_target",
        "risk_normalization_mode",
        "target_gross_exposure",
        "allow_releverage",
        "asset_class_weight_caps",
        "cluster_weight_caps",
        "effective_n_floor",
        "top_gross_share_cap_n",
        "top_gross_share_cap",
        "turnover_friction",
        "no_trade_band",
        "execution_mode",
        "turnover_budget",
        "benefit_scale",
        "min_trade_utility",
        "uncertainty_aversion",
        "risk_aversion",
        "partial_fill_enabled",
        "market_impact_bps",
        "fee_bps",
        "constraint_stages",
        "funding_bps_per_step",
        "borrow_fee_bps_per_step",
        "sleeve_count",
        "sleeves",
        "subject_set",
        "base_currency",
        "trading_calendar",
        "benchmark_id",
    )
    parts = [
        f"{key}={fields[key]}"
        for key in ordered_keys
        if key in fields
    ]
    universe_policy_text = _format_universe_policy_fields(universe_policy_fields)
    if universe_policy_text:
        parts.append(f"universe_policy=[{universe_policy_text}]")
    if subject_set_facts:
        parts.append(f"summary=[{subject_set_facts}]")
    return " ".join(parts)



def resolve_report_strategy_context(
    store: EvaluationReportStrategyContextReadPort,
    *,
    report_state,
) -> dict[str, str]:
    report = report_state.report if hasattr(report_state, "report") else report_state
    contexts: dict[str, str] = {}
    for task_result in report.task_results:
        strategy_id = task_result.strategy_id
        if strategy_id in contexts:
            continue
        if task_result.strategy_contract_fields:
            contexts[strategy_id] = format_report_strategy_contract_fields(
                task_result.strategy_contract_fields,
                subject_set_facts=task_result.subject_set_facts,
                universe_policy_fields=task_result.universe_policy_fields,
            )
            continue
        strategy_state = store.get_trading_strategy(strategy_id)
        if strategy_state is None:
            continue
        trading_strategy = strategy_state.trading_strategy
        subject_set_facts = None
        universe_policy_fields = None
        subject_set_id = trading_strategy.subject_set_id
        if subject_set_id is not None:
            subject_set_state = store.get_subject_set(subject_set_id)
            if subject_set_state is not None:
                validate_subject_set_universe_contract(subject_set_state.definition)
                subject_set_facts = format_subject_set_facts(subject_set_state.definition)
                universe_policy_fields = (
                    subject_set_state.definition.universe_policy.to_document()
                )
        contexts[strategy_id] = format_report_strategy_contract_fields(
            build_report_strategy_contract_fields(
                trading_strategy,
                subject_set=(
                    None if subject_set_id is None or subject_set_state is None else subject_set_state.definition
                ),
            ),
            subject_set_facts=subject_set_facts,
            universe_policy_fields=universe_policy_fields,
        )
    return contexts


build_report_evaluation_task_contract_fields = (
    build_report_evaluation_task_contract_fields
)
