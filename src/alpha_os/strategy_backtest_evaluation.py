from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev
from math import sqrt

import pandas as pd

from .decision_backtest import (
    DecisionBacktestResult,
)
from .portfolio_construction_config import PortfolioConstructionSpec
from .portfolio_decision import SubjectSet
from .portfolio_concentration import concentration_snapshot, top_n_gross_share
from .evaluation_result import EvaluationMetricGroupResult, EvaluationFailureFinding, EvaluationFailureFindingGroup
from .scoring import numerai_corr


_CONCENTRATION_MIN_ABS_WEIGHT = 0.001


@dataclass(frozen=True)
class StrategyBacktestRangeSummary:
    label: str
    portfolio_target_return_corr: float
    decision_net_return: float
    decision_drawdown: float
    decision_turnover: float
    decision_gross_leverage_exposure: float
    decision_net_leverage_exposure: float
    decision_long_leverage_exposure: float
    decision_short_leverage_exposure: float
    decision_gross_notional_exposure: float
    decision_net_notional_exposure: float
    decision_long_notional_exposure: float
    decision_short_notional_exposure: float
    decision_traded_notional: float
    decision_cost_notional: float
    decision_funding_cost_notional: float
    decision_borrow_cost_notional: float
    decision_roll_cost_notional: float
    optimizer_backend: str
    optimizer_status: str
    optimizer_fallback_reason: str
    mean_desired_turnover: float
    mean_executed_turnover: float
    mean_expected_execution_cost: float
    cost_to_gross_pnl: float
    execution_cost_to_gross_pnl: float
    total_execution_cost_notional: float
    top_cost_subjects: str
    top_cost_clusters: str
    mean_signal_abs_change: float
    mean_signal_sign_flip_rate: float
    mean_desired_weight_change: float
    mean_effective_n: float
    mean_active_position_count: float
    mean_top1_gross_share: float
    mean_top3_gross_share: float
    mean_top5_gross_share: float
    mean_top_intent_gross_share: float
    max_subject_gross_share: float
    max_cluster_gross_share: float
    step_count: int


def build_evaluation_metric_group_results_from_range_summaries(
    *,
    source: str,
    range_summaries: list[StrategyBacktestRangeSummary],
    all_step_net_returns: list[float],
    portfolio_construction: PortfolioConstructionSpec,
) -> tuple[dict[str, EvaluationMetricGroupResult], tuple[EvaluationFailureFindingGroup, ...]]:
    portfolio_target_return_alignment_metric_group_result = EvaluationMetricGroupResult(
        metric_group_name="portfolio_target_return_alignment",
        source=source,
        metrics={
            "mean_range_portfolio_target_return_corr": round(
                _mean([item.portfolio_target_return_corr for item in range_summaries]),
                6,
            ),
            "best_range_portfolio_target_return_corr": round(
                max(
                    (
                        item.portfolio_target_return_corr
                        for item in range_summaries
                    ),
                    default=0.0,
                ),
                6,
            ),
            "worst_range_portfolio_target_return_corr": round(
                min(
                    (
                        item.portfolio_target_return_corr
                        for item in range_summaries
                    ),
                    default=0.0,
                ),
                6,
            ),
        },
    )
    decision_metric_group_result = EvaluationMetricGroupResult(
        metric_group_name="decision_quality",
        source=source,
        metrics={
            "mean_decision_net_return": round(
                _mean([item.decision_net_return for item in range_summaries]),
                6,
            ),
            "best_decision_net_return": round(
                max((item.decision_net_return for item in range_summaries), default=0.0),
                6,
            ),
            "mean_decision_drawdown": round(
                _mean([item.decision_drawdown for item in range_summaries]),
                6,
            ),
            "mean_decision_turnover": round(
                _mean([item.decision_turnover for item in range_summaries]),
                6,
            ),
            "mean_decision_gross_leverage_exposure": round(
                _mean(
                    [item.decision_gross_leverage_exposure for item in range_summaries]
                ),
                6,
            ),
            "mean_decision_net_leverage_exposure": round(
                _mean(
                    [item.decision_net_leverage_exposure for item in range_summaries]
                ),
                6,
            ),
            "mean_decision_long_leverage_exposure": round(
                _mean(
                    [item.decision_long_leverage_exposure for item in range_summaries]
                ),
                6,
            ),
            "mean_decision_short_leverage_exposure": round(
                _mean(
                    [item.decision_short_leverage_exposure for item in range_summaries]
                ),
                6,
            ),
            "mean_decision_gross_notional_exposure": round(
                _mean(
                    [item.decision_gross_notional_exposure for item in range_summaries]
                ),
                6,
            ),
            "mean_decision_net_notional_exposure": round(
                _mean(
                    [item.decision_net_notional_exposure for item in range_summaries]
                ),
                6,
            ),
            "mean_decision_long_notional_exposure": round(
                _mean(
                    [item.decision_long_notional_exposure for item in range_summaries]
                ),
                6,
            ),
            "mean_decision_short_notional_exposure": round(
                _mean(
                    [item.decision_short_notional_exposure for item in range_summaries]
                ),
                6,
            ),
            "mean_decision_traded_notional": round(
                _mean([item.decision_traded_notional for item in range_summaries]),
                6,
            ),
            "total_decision_cost_notional": round(
                sum(item.decision_cost_notional for item in range_summaries),
                6,
            ),
            "total_decision_funding_cost_notional": round(
                sum(item.decision_funding_cost_notional for item in range_summaries),
                6,
            ),
            "total_decision_borrow_cost_notional": round(
                sum(item.decision_borrow_cost_notional for item in range_summaries),
                6,
            ),
            "total_decision_roll_cost_notional": round(
                sum(item.decision_roll_cost_notional for item in range_summaries),
                6,
            ),
            "mean_decision_step_count": round(
                _mean([float(item.step_count) for item in range_summaries]),
                6,
            ),
            "total_decision_step_count": int(len(all_step_net_returns)),
            "mean_step_net_return": round(_mean(all_step_net_returns), 6),
            "step_net_return_std": round(_std(all_step_net_returns), 6),
            "pooled_step_max_drawdown": round(
                _max_drawdown_from_step_returns(all_step_net_returns),
                6,
            ),
            "annualized_step_sharpe": round(
                _annualized_sharpe(all_step_net_returns),
                6,
            ),
        },
    )
    execution_trace_metric_group_result = EvaluationMetricGroupResult(
        metric_group_name="execution_trace",
        source=source,
        metrics={
            "mean_desired_turnover": round(
                _mean([item.mean_desired_turnover for item in range_summaries]),
                6,
            ),
            "mean_executed_turnover": round(
                _mean([item.mean_executed_turnover for item in range_summaries]),
                6,
            ),
            "mean_expected_execution_cost": round(
                _mean([item.mean_expected_execution_cost for item in range_summaries]),
                6,
            ),
        },
    )
    cost_drag_metric_group_result = EvaluationMetricGroupResult(
        metric_group_name="cost_drag",
        source=source,
        metrics={
            "cost_to_gross_pnl": round(
                _mean([item.cost_to_gross_pnl for item in range_summaries]),
                6,
            ),
            "execution_cost_to_gross_pnl": round(
                _mean(
                    [item.execution_cost_to_gross_pnl for item in range_summaries]
                ),
                6,
            ),
            "total_execution_cost_notional": round(
                _mean(
                    [item.total_execution_cost_notional for item in range_summaries]
                ),
                6,
            ),
            "top_cost_subjects": ";".join(
                item.top_cost_subjects for item in range_summaries if item.top_cost_subjects
            ),
            "top_cost_clusters": ";".join(
                item.top_cost_clusters for item in range_summaries if item.top_cost_clusters
            ),
        },
    )
    signal_churn_metric_group_result = EvaluationMetricGroupResult(
        metric_group_name="signal_churn",
        source=source,
        metrics={
            "mean_signal_abs_change": round(
                _mean([item.mean_signal_abs_change for item in range_summaries]),
                6,
            ),
            "mean_signal_sign_flip_rate": round(
                _mean([item.mean_signal_sign_flip_rate for item in range_summaries]),
                6,
            ),
            "mean_desired_weight_change": round(
                _mean([item.mean_desired_weight_change for item in range_summaries]),
                6,
            ),
        },
    )
    concentration_metric_group_result = EvaluationMetricGroupResult(
        metric_group_name="portfolio_concentration",
        source=source,
        metrics={
            "mean_effective_n": round(
                _mean([item.mean_effective_n for item in range_summaries]),
                6,
            ),
            "mean_active_position_count": round(
                _mean([item.mean_active_position_count for item in range_summaries]),
                6,
            ),
            "mean_top1_gross_share": round(
                _mean([item.mean_top1_gross_share for item in range_summaries]),
                6,
            ),
            "mean_top3_gross_share": round(
                _mean([item.mean_top3_gross_share for item in range_summaries]),
                6,
            ),
            "mean_top5_gross_share": round(
                _mean([item.mean_top5_gross_share for item in range_summaries]),
                6,
            ),
            "mean_top_intent_gross_share": round(
                _mean([item.mean_top_intent_gross_share for item in range_summaries]),
                6,
            ),
            "max_subject_gross_share": round(
                max((item.max_subject_gross_share for item in range_summaries), default=0.0),
                6,
            ),
            "max_cluster_gross_share": round(
                max((item.max_cluster_gross_share for item in range_summaries), default=0.0),
                6,
            ),
            "effective_n_floor": (
                "-"
                if portfolio_construction.effective_n_floor is None
                else portfolio_construction.effective_n_floor
            ),
            "top_gross_share_cap_n": (
                "-"
                if portfolio_construction.top_gross_share_cap_n is None
                else portfolio_construction.top_gross_share_cap_n
            ),
            "top_gross_share_cap": (
                "-"
                if portfolio_construction.top_gross_share_cap is None
                else portfolio_construction.top_gross_share_cap
            ),
        },
    )
    robustness_metric_group_result = EvaluationMetricGroupResult(
        metric_group_name="robustness",
        source=source,
        metrics={
            "range_count": len(range_summaries),
            "portfolio_target_return_corr_std": round(
                _std([item.portfolio_target_return_corr for item in range_summaries]),
                6,
            ),
            "decision_net_return_std": round(
                _std([item.decision_net_return for item in range_summaries]),
                6,
            ),
            "decision_negative_fraction": round(
                0.0
                if not range_summaries
                else sum(1 for item in range_summaries if item.decision_net_return <= 0.0)
                / float(len(range_summaries)),
                6,
            ),
            "worst_decision_net_return": round(
                min((item.decision_net_return for item in range_summaries), default=0.0),
                6,
            ),
        },
    )
    return (
        {
            portfolio_target_return_alignment_metric_group_result.metric_group_name: (
                portfolio_target_return_alignment_metric_group_result
            ),
            decision_metric_group_result.metric_group_name: decision_metric_group_result,
            execution_trace_metric_group_result.metric_group_name: execution_trace_metric_group_result,
            cost_drag_metric_group_result.metric_group_name: cost_drag_metric_group_result,
            signal_churn_metric_group_result.metric_group_name: signal_churn_metric_group_result,
            concentration_metric_group_result.metric_group_name: concentration_metric_group_result,
            robustness_metric_group_result.metric_group_name: robustness_metric_group_result,
        },
        _failure_finding_groups(
            range_summaries,
            portfolio_construction=portfolio_construction,
            source=source,
        ),
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(pstdev(values))


def _annualized_sharpe(
    values: list[float],
    *,
    periods_per_year: int = 252,
) -> float:
    if len(values) < 2:
        return 0.0
    mean_return = _mean(values)
    std_return = _std(values)
    if std_return <= 0.0:
        return 0.0
    return float((mean_return / std_return) * sqrt(float(periods_per_year)))


def _max_drawdown_from_step_returns(values: list[float]) -> float:
    if not values:
        return 0.0
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in values:
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        if peak > 0.0:
            drawdown = max(drawdown, 1.0 - (equity / peak))
    return float(drawdown)


def _range_summary_from_backtest_result(
    range_label: str,
    backtest_result: DecisionBacktestResult,
    *,
    portfolio_construction: PortfolioConstructionSpec,
    subject_set: SubjectSet | None,
) -> StrategyBacktestRangeSummary:
    concentration = _portfolio_concentration_from_backtest(
        backtest_result,
        subject_set=subject_set,
        min_abs_weight=_CONCENTRATION_MIN_ABS_WEIGHT,
        top_intent_n=portfolio_construction.top_gross_share_cap_n,
    )
    execution_impact = _execution_trace_from_backtest(backtest_result)
    cost_drag = _cost_drag_from_backtest(backtest_result, subject_set=subject_set)
    signal_churn = _signal_churn_from_backtest(backtest_result)
    optimizer_diagnostics = _optimizer_diagnostics_from_backtest(backtest_result)
    return StrategyBacktestRangeSummary(
        label=range_label,
        portfolio_target_return_corr=_portfolio_target_return_corr(backtest_result),
        decision_net_return=backtest_result.net_return_total,
        decision_drawdown=backtest_result.max_drawdown,
        decision_turnover=backtest_result.mean_turnover,
        decision_gross_leverage_exposure=backtest_result.mean_gross_leverage_exposure,
        decision_net_leverage_exposure=backtest_result.mean_net_leverage_exposure,
        decision_long_leverage_exposure=backtest_result.mean_long_leverage_exposure,
        decision_short_leverage_exposure=backtest_result.mean_short_leverage_exposure,
        decision_gross_notional_exposure=backtest_result.mean_gross_notional_exposure,
        decision_net_notional_exposure=backtest_result.mean_net_notional_exposure,
        decision_long_notional_exposure=backtest_result.mean_long_notional_exposure,
        decision_short_notional_exposure=backtest_result.mean_short_notional_exposure,
        decision_traded_notional=backtest_result.mean_traded_notional,
        decision_cost_notional=backtest_result.cost_notional_total,
        decision_funding_cost_notional=backtest_result.funding_cost_notional_total,
        decision_borrow_cost_notional=backtest_result.borrow_cost_notional_total,
        decision_roll_cost_notional=backtest_result.roll_cost_notional_total,
        optimizer_backend=str(optimizer_diagnostics["optimizer_backend"]),
        optimizer_status=str(optimizer_diagnostics["optimizer_status"]),
        optimizer_fallback_reason=str(
            optimizer_diagnostics["optimizer_fallback_reason"]
        ),
        mean_desired_turnover=execution_impact["mean_desired_turnover"],
        mean_executed_turnover=execution_impact["mean_executed_turnover"],
        mean_expected_execution_cost=execution_impact[
            "mean_expected_execution_cost"
        ],
        cost_to_gross_pnl=float(cost_drag["cost_to_gross_pnl"]),
        execution_cost_to_gross_pnl=float(cost_drag["execution_cost_to_gross_pnl"]),
        total_execution_cost_notional=float(cost_drag["total_execution_cost_notional"]),
        top_cost_subjects=str(cost_drag["top_cost_subjects"]),
        top_cost_clusters=str(cost_drag["top_cost_clusters"]),
        mean_signal_abs_change=signal_churn["mean_signal_abs_change"],
        mean_signal_sign_flip_rate=signal_churn["mean_signal_sign_flip_rate"],
        mean_desired_weight_change=signal_churn["mean_desired_weight_change"],
        mean_effective_n=concentration["mean_effective_n"],
        mean_active_position_count=concentration["mean_active_position_count"],
        mean_top1_gross_share=concentration["mean_top1_gross_share"],
        mean_top3_gross_share=concentration["mean_top3_gross_share"],
        mean_top5_gross_share=concentration["mean_top5_gross_share"],
        mean_top_intent_gross_share=concentration["mean_top_intent_gross_share"],
        max_subject_gross_share=concentration["max_subject_gross_share"],
        max_cluster_gross_share=concentration["max_cluster_gross_share"],
        step_count=len(backtest_result.steps),
    )


def _optimizer_diagnostics_from_backtest(
    result: DecisionBacktestResult,
) -> dict[str, str]:
    diagnostics = [
        step.sizing_diagnostics
        for step in result.steps
        if step.sizing_diagnostics.backend_id != "-"
    ]
    return {
        "optimizer_backend": _ordered_unique_join(
            [item.backend_id for item in diagnostics]
        ),
        "optimizer_status": _ordered_unique_join(
            [item.status for item in diagnostics if item.status != "-"]
        ),
        "optimizer_fallback_reason": _ordered_unique_join(
            [
                item.fallback_reason
                for item in diagnostics
                if item.fallback_reason is not None
            ]
        ),
    }


def _ordered_unique_join(values: list[str]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ";".join(ordered) if ordered else "-"


def _execution_trace_from_backtest(result: DecisionBacktestResult) -> dict[str, float]:
    traces = [
        step.execution_trace
        for step in result.steps
        if getattr(step, "execution_trace", None) is not None
    ]
    return {
        "mean_desired_turnover": _mean(
            [float(item.desired_turnover) for item in traces]
        ),
        "mean_executed_turnover": _mean(
            [float(item.executed_turnover) for item in traces]
        ),
        "mean_expected_execution_cost": _mean(
            [float(item.expected_execution_cost) for item in traces]
        ),
    }


def _cost_drag_from_backtest(
    result: DecisionBacktestResult,
    *,
    subject_set: SubjectSet | None,
) -> dict[str, float | str]:
    subject_costs: dict[str, float] = {}
    cluster_costs: dict[str, float] = {}
    total_gross_pnl = 0.0
    total_cost = 0.0
    total_execution_cost = 0.0
    cluster_by_subject = {} if subject_set is None else subject_set.cluster_by_subject
    for step in result.steps:
        for subject_step in step.subject_steps:
            total_gross_pnl += float(subject_step.gross_pnl_notional)
            total_cost += float(subject_step.cost_notional)
            total_execution_cost += float(subject_step.execution_cost_notional)
            subject_costs[subject_step.subject_id] = (
                subject_costs.get(subject_step.subject_id, 0.0)
                + float(subject_step.cost_notional)
            )
            cluster_id = cluster_by_subject.get(subject_step.subject_id, "unknown")
            cluster_costs[cluster_id] = (
                cluster_costs.get(cluster_id, 0.0) + float(subject_step.cost_notional)
            )
    denominator = abs(total_gross_pnl)
    return {
        "cost_to_gross_pnl": (
            0.0 if denominator <= 0.0 else float(total_cost / denominator)
        ),
        "execution_cost_to_gross_pnl": (
            0.0 if denominator <= 0.0 else float(total_execution_cost / denominator)
        ),
        "total_execution_cost_notional": float(total_execution_cost),
        "top_cost_subjects": _format_top_cost_items(subject_costs),
        "top_cost_clusters": _format_top_cost_items(cluster_costs),
    }


def _format_top_cost_items(values: dict[str, float], *, limit: int = 3) -> str:
    items = sorted(values.items(), key=lambda item: (-abs(item[1]), item[0]))[:limit]
    return ",".join(f"{name}={value:.6f}" for name, value in items)


def _signal_churn_from_backtest(result: DecisionBacktestResult) -> dict[str, float]:
    signal_changes: list[float] = []
    sign_flips: list[float] = []
    desired_weight_changes: list[float] = []
    previous_signals: dict[str, float] | None = None
    for step in result.steps:
        current_signals = {
            item.subject_id: float(item.signal_value) for item in step.subject_steps
        }
        if previous_signals is not None:
            for subject_id, current_signal in current_signals.items():
                previous_signal = previous_signals.get(subject_id)
                if previous_signal is None:
                    continue
                signal_changes.append(abs(current_signal - previous_signal))
                sign_flips.append(
                    1.0
                    if _signal_sign(previous_signal) != 0
                    and _signal_sign(current_signal) != 0
                    and _signal_sign(previous_signal) != _signal_sign(current_signal)
                    else 0.0
                )
        previous_signals = current_signals
        if step.execution_trace is not None:
            desired_weight_changes.extend(
                abs(item.desired_delta) for item in step.execution_trace.subjects
            )
    return {
        "mean_signal_abs_change": _mean(signal_changes),
        "mean_signal_sign_flip_rate": _mean(sign_flips),
        "mean_desired_weight_change": _mean(desired_weight_changes),
    }


def _signal_sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _portfolio_concentration_from_backtest(
    result: DecisionBacktestResult,
    *,
    subject_set: SubjectSet | None,
    min_abs_weight: float,
    top_intent_n: int | None,
) -> dict[str, float]:
    cluster_by_subject = {} if subject_set is None else subject_set.cluster_by_subject
    snapshots = []
    top_intent_shares: list[float] = []
    for step in result.steps:
        weights_by_subject = {
            item.subject_id: float(item.target_weight)
            for item in step.subject_steps
        }
        if top_intent_n is not None:
            top_intent_shares.append(
                top_n_gross_share(weights_by_subject.values(), top_n=top_intent_n)
            )
        snapshots.append(
            concentration_snapshot(
                weights_by_subject,
                cluster_by_subject=cluster_by_subject,
                min_abs_weight=min_abs_weight,
            )
        )
    return {
        "mean_effective_n": _mean([item.effective_n for item in snapshots]),
        "mean_active_position_count": _mean(
            [float(item.active_position_count) for item in snapshots]
        ),
        "mean_top1_gross_share": _mean([item.top1_gross_share for item in snapshots]),
        "mean_top3_gross_share": _mean([item.top3_gross_share for item in snapshots]),
        "mean_top5_gross_share": _mean([item.top5_gross_share for item in snapshots]),
        "mean_top_intent_gross_share": _mean(top_intent_shares),
        "max_subject_gross_share": max(
            (item.max_subject_gross_share for item in snapshots),
            default=0.0,
        ),
        "max_cluster_gross_share": max(
            (item.max_cluster_gross_share for item in snapshots),
            default=0.0,
        ),
    }


def _failure_finding_groups(
    range_summaries: list[StrategyBacktestRangeSummary],
    *,
    portfolio_construction: PortfolioConstructionSpec,
    source: str,
) -> tuple[EvaluationFailureFindingGroup, ...]:
    decision_cases: list[EvaluationFailureFinding] = []
    portfolio_target_return_alignment_cases: list[EvaluationFailureFinding] = []
    concentration_cases: list[EvaluationFailureFinding] = []
    for item in sorted(
        range_summaries,
        key=lambda row: (
            row.decision_net_return,
            row.portfolio_target_return_corr,
            row.label,
        ),
    ):
        if item.decision_net_return <= 0.0:
            decision_cases.append(
                EvaluationFailureFinding(
                    label=item.label,
                    severity=abs(float(item.decision_net_return)),
                    metrics={
                        "decision_net_return": round(item.decision_net_return, 6),
                        "decision_drawdown": round(item.decision_drawdown, 6),
                        "decision_turnover": round(item.decision_turnover, 6),
                        "step_count": item.step_count,
                    },
                )
            )
        if item.portfolio_target_return_corr <= 0.0:
            portfolio_target_return_alignment_cases.append(
                EvaluationFailureFinding(
                    label=item.label,
                    severity=abs(float(item.portfolio_target_return_corr)),
                    metrics={
                        "portfolio_target_return_corr": round(
                            item.portfolio_target_return_corr,
                            6,
                        ),
                        "decision_net_return": round(item.decision_net_return, 6),
                        "step_count": item.step_count,
                    },
                )
            )
        concentration_severity = 0.0
        if portfolio_construction.effective_n_floor is not None:
            concentration_severity = max(
                concentration_severity,
                float(portfolio_construction.effective_n_floor) - item.mean_effective_n,
            )
        if (
            portfolio_construction.top_gross_share_cap_n is not None
            and portfolio_construction.top_gross_share_cap is not None
        ):
            concentration_severity = max(
                concentration_severity,
                item.mean_top_intent_gross_share
                - float(portfolio_construction.top_gross_share_cap),
            )
        if concentration_severity > 0.0:
            concentration_cases.append(
                EvaluationFailureFinding(
                    label=item.label,
                    severity=abs(float(concentration_severity)),
                    metrics={
                        "mean_effective_n": round(item.mean_effective_n, 6),
                        "mean_top3_gross_share": round(
                            item.mean_top3_gross_share,
                            6,
                        ),
                        "mean_top_intent_gross_share": round(
                            item.mean_top_intent_gross_share,
                            6,
                        ),
                        "effective_n_floor": (
                            "-"
                            if portfolio_construction.effective_n_floor is None
                            else portfolio_construction.effective_n_floor
                        ),
                        "top_gross_share_cap": (
                            "-"
                            if portfolio_construction.top_gross_share_cap is None
                            else portfolio_construction.top_gross_share_cap
                        ),
                        "step_count": item.step_count,
                    },
                )
            )
    return (
        EvaluationFailureFindingGroup(
            metric_group_name="decision_quality",
            source=source,
            findings=tuple(decision_cases),
        ),
        EvaluationFailureFindingGroup(
            metric_group_name="portfolio_target_return_alignment",
            source=source,
            findings=tuple(portfolio_target_return_alignment_cases),
        ),
        EvaluationFailureFindingGroup(
            metric_group_name="portfolio_concentration",
            source=source,
            findings=tuple(concentration_cases),
        ),
    )


def _portfolio_target_return_corr(backtest_result: DecisionBacktestResult) -> float:
    step_corrs: list[float] = []
    for step in backtest_result.steps:
        weights = {
            item.subject_id: float(item.target_weight)
            for item in step.subject_steps
        }
        realized_returns = {
            item.subject_id: float(item.realized_return)
            for item in step.subject_steps
        }
        aligned = pd.concat(
            [
                pd.Series(weights, dtype=float),
                pd.Series(realized_returns, dtype=float),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) < 2:
            continue
        step_corrs.append(
            numerai_corr(
                aligned.iloc[:, 0].astype(float),
                aligned.iloc[:, 1].astype(float),
            )
        )
    return _mean(step_corrs)
