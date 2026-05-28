from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable
from statistics import pstdev
from math import sqrt

import pandas as pd

from .decision_backtest import (
    DependenceBacktestSeries,
    DecisionBacktestInput,
    DecisionBacktestResult,
    SubjectBacktestSeries,
    run_decision_backtest,
)
from .evaluation_cost_config import TradingEnvironment
from .evaluation_spec import EvaluationDateRange
from .portfolio_construction_config import PortfolioConstructionSpec
from .evaluation_metric_group_result_builders import (
    build_portfolio_construction_trace_metric_group_result,
    build_prediction_diagnostics_metric_group_result,
)
from .portfolio_decision import SubjectSet
from .portfolio_concentration import concentration_snapshot, top_n_gross_share
from .prediction_diagnostics import (
    PredictionDiagnostics,
    build_prediction_diagnostics,
)
from .portfolio_sizing_policy import (
    ConstrainedOptimizerSizingPolicy,
    HistoricalModelSizingPolicy,
    SignalWeightedSizingPolicy,
    SignedMeanVarianceSizingPolicy,
)
from .evaluation_result import EvaluationMetricGroupResult, EvaluationFailureFinding, EvaluationFailureFindingGroup
from .scoring import numerai_corr


_CONCENTRATION_MIN_ABS_WEIGHT = 0.001


@dataclass(frozen=True)
class StrategyBacktestRangeSummary:
    label: str
    predictive_corr: float
    prediction_hit_rate: float
    prediction_long_short_spread: float
    prediction_long_bucket_return: float
    prediction_short_bucket_return: float
    prediction_coverage: float
    prediction_positive_group_fraction: float
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
    target_vol_stage_mean_gross_delta: float
    net_target_stage_mean_net_delta: float
    top_k_stage_mean_active_count_delta: float
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
    daily_rebalance_net_return: float
    daily_rebalance_drawdown: float
    daily_rebalance_turnover: float
    equal_weight_net_return: float
    equal_weight_drawdown: float
    equal_weight_turnover: float
    equal_weight_daily_net_return: float
    equal_weight_daily_drawdown: float
    equal_weight_daily_turnover: float
    step_count: int


@dataclass(frozen=True)
class RangeBacktestDataset:
    label: str
    predictive_corr: float
    prediction_diagnostics: PredictionDiagnostics
    subject_series: tuple[SubjectBacktestSeries, ...]
    dependence_series: tuple[DependenceBacktestSeries, ...] = ()


@dataclass(frozen=True)
class RangeBacktestVariantResults:
    selected: DecisionBacktestResult
    daily_rebalance: DecisionBacktestResult
    equal_weight: DecisionBacktestResult
    equal_weight_daily: DecisionBacktestResult


@dataclass(frozen=True)
class EvaluationTraceRangeResult:
    range_label: str
    result: DecisionBacktestResult


@dataclass(frozen=True)
class StrategyEvaluationResult:
    metric_group_results_by_name: dict[str, EvaluationMetricGroupResult]
    failure_finding_groups: tuple[EvaluationFailureFindingGroup, ...]
    selected_trace_results: tuple[EvaluationTraceRangeResult, ...] = ()

    def __iter__(self):
        yield self.metric_group_results_by_name
        yield self.failure_finding_groups


@dataclass(frozen=True)
class RangeBacktestEvaluationLoopResult:
    range_summaries: tuple[StrategyBacktestRangeSummary, ...]
    selected_trace_results: tuple[EvaluationTraceRangeResult, ...]
    all_step_net_returns: tuple[float, ...]


def evaluate_range_backtest_dataset_builder(
    *,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
    build_dataset_for_range: Callable[[EvaluationDateRange], RangeBacktestDataset | None],
    subject_set_id: str | None,
    subject_set: SubjectSet | None,
    target_id: str,
    portfolio_construction: PortfolioConstructionSpec,
    trading_environment: TradingEnvironment,
    top_k: int | None,
) -> RangeBacktestEvaluationLoopResult:
    range_summaries: list[StrategyBacktestRangeSummary] = []
    selected_trace_results: list[EvaluationTraceRangeResult] = []
    all_step_net_returns: list[float] = []
    for date_range in evaluation_date_ranges:
        dataset = build_dataset_for_range(date_range)
        if dataset is None:
            continue
        variant_results = evaluate_range_backtest_variants(
            subject_set_id=subject_set_id,
            subject_set=subject_set,
            target_id=target_id,
            dataset=dataset,
            portfolio_construction=portfolio_construction,
            trading_environment=trading_environment,
            top_k=top_k,
        )
        all_step_net_returns.extend(
            float(step.net_return) for step in variant_results.selected.steps
        )
        selected_trace_results.append(
            EvaluationTraceRangeResult(
                range_label=date_range.label,
                result=variant_results.selected,
            )
        )
        range_summaries.append(
            _range_summary_from_variant_results(
                dataset,
                variant_results,
                portfolio_construction=portfolio_construction,
                subject_set=subject_set,
            )
        )
    return RangeBacktestEvaluationLoopResult(
        range_summaries=tuple(range_summaries),
        selected_trace_results=tuple(selected_trace_results),
        all_step_net_returns=tuple(all_step_net_returns),
    )


def build_direct_range_backtest_dataset_for_range(
    *,
    date_range: EvaluationDateRange,
    subject_return_series_by_subject: dict[str, pd.Series],
    signal_series_by_subject: dict[str, pd.Series] | None,
    funding_cost_bps_series_by_subject: dict[str, pd.Series] | None,
    borrow_fee_bps_series_by_subject: dict[str, pd.Series] | None,
    roll_cost_bps_series_by_subject: dict[str, pd.Series] | None,
    contract_multiplier_by_subject: dict[str, float] | None,
    signal_value: float,
) -> RangeBacktestDataset | None:
    return build_direct_range_backtest_dataset(
        date_range=date_range,
        subject_return_series_by_subject=subject_return_series_by_subject,
        signal_series_by_subject=signal_series_by_subject,
        funding_cost_bps_series_by_subject=funding_cost_bps_series_by_subject,
        borrow_fee_bps_series_by_subject=borrow_fee_bps_series_by_subject,
        roll_cost_bps_series_by_subject=roll_cost_bps_series_by_subject,
        contract_multiplier_by_subject=contract_multiplier_by_subject,
        signal_value=signal_value,
    )


def build_direct_strategy_evaluation_metric_group_results(
    *,
    subject_return_series_by_subject: dict[str, pd.Series],
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
    target_id: str,
    subject_set_id: str | None,
    subject_set: SubjectSet | None = None,
    signal_series_by_subject: dict[str, pd.Series] | None = None,
    funding_cost_bps_series_by_subject: dict[str, pd.Series] | None = None,
    borrow_fee_bps_series_by_subject: dict[str, pd.Series] | None = None,
    roll_cost_bps_series_by_subject: dict[str, pd.Series] | None = None,
    contract_multiplier_by_subject: dict[str, float] | None = None,
    portfolio_construction: PortfolioConstructionSpec = PortfolioConstructionSpec(),
    trading_environment: TradingEnvironment = TradingEnvironment(),
    top_k: int | None = None,
    signal_value: float = 1.0,
) -> StrategyEvaluationResult:
    def build_dataset_for_range(
        date_range: EvaluationDateRange,
    ) -> RangeBacktestDataset | None:
        return build_direct_range_backtest_dataset_for_range(
            date_range=date_range,
            subject_return_series_by_subject=subject_return_series_by_subject,
            signal_series_by_subject=signal_series_by_subject,
            funding_cost_bps_series_by_subject=funding_cost_bps_series_by_subject,
            borrow_fee_bps_series_by_subject=borrow_fee_bps_series_by_subject,
            roll_cost_bps_series_by_subject=roll_cost_bps_series_by_subject,
            contract_multiplier_by_subject=contract_multiplier_by_subject,
            signal_value=signal_value,
        )

    loop_result = evaluate_range_backtest_dataset_builder(
        evaluation_date_ranges=evaluation_date_ranges,
        build_dataset_for_range=build_dataset_for_range,
        subject_set_id=subject_set_id,
        subject_set=subject_set,
        target_id=target_id,
        portfolio_construction=portfolio_construction,
        trading_environment=trading_environment,
        top_k=top_k,
    )
    (
        metric_group_results_by_name,
        failure_finding_groups,
    ) = build_evaluation_metric_group_results_from_range_summaries(
        source="direct_plan",
        range_summaries=list(loop_result.range_summaries),
        all_step_net_returns=list(loop_result.all_step_net_returns),
        portfolio_construction=portfolio_construction,
    )
    return StrategyEvaluationResult(
        metric_group_results_by_name=metric_group_results_by_name,
        failure_finding_groups=failure_finding_groups,
        selected_trace_results=loop_result.selected_trace_results,
    )

def build_evaluation_metric_group_results_from_range_summaries(
    *,
    source: str,
    range_summaries: list[StrategyBacktestRangeSummary],
    all_step_net_returns: list[float],
    portfolio_construction: PortfolioConstructionSpec,
) -> tuple[dict[str, EvaluationMetricGroupResult], tuple[EvaluationFailureFindingGroup, ...]]:
    prediction_diagnostics_metric_group_result = (
        build_prediction_diagnostics_metric_group_result(
            source=source,
            range_summaries=range_summaries,
            mean=_mean,
        )
    )
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
    construction_trace_metric_group_result = (
        build_portfolio_construction_trace_metric_group_result(
            source=source,
            range_summaries=range_summaries,
            mean=_mean,
        )
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
    sizing_policy_metric_group_result = EvaluationMetricGroupResult(
        metric_group_name="sizing_policy_quality",
        source=source,
        metrics={
            "selected_sizing_method": portfolio_construction.sizing_method,
            "optimizer_backend": _ordered_unique_join(
                [item.optimizer_backend for item in range_summaries]
            ),
            "optimizer_status": _ordered_unique_join(
                [item.optimizer_status for item in range_summaries]
            ),
            "optimizer_fallback_reason": _ordered_unique_join(
                [
                    item.optimizer_fallback_reason
                    for item in range_summaries
                    if item.optimizer_fallback_reason != "-"
                ]
            ),
            "mean_equal_weight_decision_net_return": round(
                _mean([item.equal_weight_net_return for item in range_summaries]),
                6,
            ),
            "mean_equal_weight_daily_decision_net_return": round(
                _mean([item.equal_weight_daily_net_return for item in range_summaries]),
                6,
            ),
            "mean_selected_vs_equal_weight_decision_net_return_edge": round(
                _mean(
                    [
                        item.decision_net_return - item.equal_weight_net_return
                        for item in range_summaries
                    ]
                ),
                6,
            ),
            "best_selected_vs_equal_weight_decision_net_return_edge": round(
                max(
                    (
                        item.decision_net_return - item.equal_weight_net_return
                        for item in range_summaries
                    ),
                    default=0.0,
                ),
                6,
            ),
            "worst_selected_vs_equal_weight_decision_net_return_edge": round(
                min(
                    (
                        item.decision_net_return - item.equal_weight_net_return
                        for item in range_summaries
                    ),
                    default=0.0,
                ),
                6,
            ),
            "mean_daily_signal_weighted_vs_equal_weight_decision_net_return_edge": round(
                _mean(
                    [
                        item.daily_rebalance_net_return
                        - item.equal_weight_daily_net_return
                        for item in range_summaries
                    ]
                ),
                6,
            ),
            "mean_selected_vs_equal_weight_drawdown_edge": round(
                _mean(
                    [
                        item.equal_weight_drawdown - item.decision_drawdown
                        for item in range_summaries
                    ]
                ),
                6,
            ),
            "mean_selected_vs_equal_weight_turnover_edge": round(
                _mean(
                    [
                        item.equal_weight_turnover - item.decision_turnover
                        for item in range_summaries
                    ]
                ),
                6,
            ),
        },
    )
    rebalance_policy_metric_group_result = EvaluationMetricGroupResult(
        metric_group_name="rebalance_policy_quality",
        source=source,
        metrics={
            "selected_rebalance_interval_steps": int(
                portfolio_construction.rebalance_interval_steps
            ),
            "daily_rebalance_reference": (
                "selected_reused_for_expensive_optimizer"
                if _uses_expensive_sizing_optimizer(portfolio_construction)
                else "computed"
            ),
            "mean_selected_vs_daily_rebalance_net_return_edge": round(
                _mean(
                    [
                        item.decision_net_return - item.daily_rebalance_net_return
                        for item in range_summaries
                    ]
                ),
                6,
            ),
            "best_selected_vs_daily_rebalance_net_return_edge": round(
                max(
                    (
                        item.decision_net_return - item.daily_rebalance_net_return
                        for item in range_summaries
                    ),
                    default=0.0,
                ),
                6,
            ),
            "worst_selected_vs_daily_rebalance_net_return_edge": round(
                min(
                    (
                        item.decision_net_return - item.daily_rebalance_net_return
                        for item in range_summaries
                    ),
                    default=0.0,
                ),
                6,
            ),
            "mean_selected_vs_daily_rebalance_turnover_savings": round(
                _mean(
                    [
                        item.daily_rebalance_turnover - item.decision_turnover
                        for item in range_summaries
                    ]
                ),
                6,
            ),
            "mean_equal_weight_vs_daily_rebalance_net_return_edge": round(
                _mean(
                    [
                        item.equal_weight_net_return - item.equal_weight_daily_net_return
                        for item in range_summaries
                    ]
                ),
                6,
            ),
            "mean_equal_weight_vs_daily_rebalance_turnover_savings": round(
                _mean(
                    [
                        item.equal_weight_daily_turnover - item.equal_weight_turnover
                        for item in range_summaries
                    ]
                ),
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
            "predictive_corr_std": round(
                _std([item.predictive_corr for item in range_summaries]),
                6,
            ),
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
            prediction_diagnostics_metric_group_result.metric_group_name: (
                prediction_diagnostics_metric_group_result
            ),
            portfolio_target_return_alignment_metric_group_result.metric_group_name: (
                portfolio_target_return_alignment_metric_group_result
            ),
            decision_metric_group_result.metric_group_name: decision_metric_group_result,
            construction_trace_metric_group_result.metric_group_name: (
                construction_trace_metric_group_result
            ),
            execution_trace_metric_group_result.metric_group_name: execution_trace_metric_group_result,
            cost_drag_metric_group_result.metric_group_name: cost_drag_metric_group_result,
            signal_churn_metric_group_result.metric_group_name: signal_churn_metric_group_result,
            sizing_policy_metric_group_result.metric_group_name: sizing_policy_metric_group_result,
            rebalance_policy_metric_group_result.metric_group_name: rebalance_policy_metric_group_result,
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


def _uses_expensive_sizing_optimizer(
    portfolio_construction: PortfolioConstructionSpec,
) -> bool:
    return portfolio_construction.sizing_method == "signed_mean_variance"


def _portfolio_sizing_policy_from_config(config: PortfolioConstructionSpec):
    if config.sizing_method == "signal_weighted":
        return SignalWeightedSizingPolicy()
    if config.sizing_method == "constrained_signal_weighted":
        return ConstrainedOptimizerSizingPolicy()
    if config.sizing_method == "signed_mean_variance":
        return SignedMeanVarianceSizingPolicy()
    if config.sizing_method in {
        "equal_weight",
        "minimum_variance",
        "risk_budgeting",
        "hierarchical_risk_parity",
        "conviction_adjusted_hierarchical_risk_parity",
        "diversified_risk_budget",
    }:
        return HistoricalModelSizingPolicy(
            model_type=config.sizing_method,
            effective_n_floor=config.effective_n_floor,
            top_gross_share_cap_n=config.top_gross_share_cap_n,
            top_gross_share_cap=config.top_gross_share_cap,
        )
    raise ValueError(
        "unsupported decision backtest config: "
        f"{config.sizing_method}"
    )


def _portfolio_construction_variant(
    portfolio_construction: PortfolioConstructionSpec,
    *,
    sizing_method: str | None = None,
    rebalance_interval_steps: int | None = None,
) -> PortfolioConstructionSpec:
    return replace(
        portfolio_construction,
        sizing_method=(
            portfolio_construction.sizing_method
            if sizing_method is None
            else sizing_method
        ),
        rebalance_interval_steps=(
            portfolio_construction.rebalance_interval_steps
            if rebalance_interval_steps is None
            else rebalance_interval_steps
        ),
    )


def _run_backtest_variant(
    *,
    subject_set_id: str | None,
    subject_set: SubjectSet | None = None,
    target_id: str,
    subject_series: tuple[SubjectBacktestSeries, ...],
    dependence_series: tuple[DependenceBacktestSeries, ...],
    portfolio_construction: PortfolioConstructionSpec,
    trading_environment: TradingEnvironment,
    top_k: int | None,
) -> DecisionBacktestResult:
    return run_decision_backtest(
        DecisionBacktestInput(
            portfolio_id="evaluation",
            subject_set_id=subject_set_id,
            portfolio_construction=portfolio_construction,
            asset_class_by_subject=(
                {} if subject_set is None else subject_set.asset_class_by_subject
            ),
            cluster_by_subject=(
                {} if subject_set is None else subject_set.cluster_by_subject
            ),
            asset_class_weight_caps=dict(portfolio_construction.asset_class_weight_caps),
            cluster_weight_caps=dict(portfolio_construction.cluster_weight_caps),
            target_id=target_id,
            subject_series=subject_series,
            dependence_series=dependence_series,
            gross_exposure_cap=portfolio_construction.gross_exposure_cap,
            target_vol=portfolio_construction.target_vol,
            gross_leverage_cap=portfolio_construction.gross_leverage_cap,
            net_exposure_target=portfolio_construction.net_exposure_target,
            turnover_cost_rate=trading_environment.turnover_cost_rate,
            market_impact_bps=trading_environment.market_impact_bps,
            fee_bps=trading_environment.fee_bps,
            bid_ask_spread_bps=trading_environment.bid_ask_spread_bps,
            funding_bps_per_step=trading_environment.funding_bps_per_step,
            borrow_fee_bps_per_step=trading_environment.borrow_fee_bps_per_step,
            rebalance_interval_steps=portfolio_construction.rebalance_interval_steps,
            long_only=portfolio_construction.long_only,
            direction_mode=portfolio_construction.direction_mode,
            top_k=top_k,
            active_weight_budget=portfolio_construction.active_weight_budget,
            historical_return_lookback_steps=_historical_return_lookback_steps(
                portfolio_construction
            ),
            subject_metadata_by_subject=_subject_metadata_by_subject(subject_set),
        ),
        sizing_policy=_portfolio_sizing_policy_from_config(portfolio_construction),
    )


def _historical_return_lookback_steps(
    portfolio_construction: PortfolioConstructionSpec,
) -> int | None:
    sizing_method = portfolio_construction.sizing_method
    if sizing_method == "equal_weight":
        return 0
    if sizing_method in {
        "signed_mean_variance",
        "conviction_adjusted_hierarchical_risk_parity",
    }:
        return 756
    return None


def _subject_metadata_by_subject(
    subject_set: SubjectSet | None,
) -> dict[str, dict[str, str]]:
    if subject_set is None:
        return {}
    metadata: dict[str, dict[str, str]] = {}
    for subject_id in subject_set.subject_ids:
        instrument = subject_set.instrument_for_subject(subject_id)
        if instrument is None:
            metadata[subject_id] = {}
            continue
        values = {
            "instrument_type": instrument.instrument_type,
            "asset_class": instrument.asset_class,
            "region": instrument.region,
            "cluster": instrument.cluster,
        }
        metadata[subject_id] = {
            key: value
            for key, value in values.items()
            if value is not None
        }
    return metadata


def build_direct_range_backtest_dataset(
    *,
    date_range: EvaluationDateRange,
    subject_return_series_by_subject: dict[str, pd.Series],
    signal_series_by_subject: dict[str, pd.Series] | None,
    funding_cost_bps_series_by_subject: dict[str, pd.Series] | None,
    borrow_fee_bps_series_by_subject: dict[str, pd.Series] | None,
    roll_cost_bps_series_by_subject: dict[str, pd.Series] | None,
    contract_multiplier_by_subject: dict[str, float] | None,
    signal_value: float,
) -> RangeBacktestDataset | None:
    subject_series: list[SubjectBacktestSeries] = []
    for subject_id, full_returns in sorted(subject_return_series_by_subject.items()):
        range_returns = full_returns.loc[
            (full_returns.index >= date_range.start_date)
            & (full_returns.index <= date_range.end_date)
        ].dropna()
        if range_returns.empty:
            continue
        if signal_series_by_subject is None:
            signal_series = pd.Series(
                float(signal_value),
                index=range_returns.index,
                dtype=float,
            )
        else:
            subject_signal_series = signal_series_by_subject.get(subject_id)
            if subject_signal_series is None:
                signal_series = pd.Series(
                    0.0,
                    index=range_returns.index,
                    dtype=float,
                )
            else:
                signal_series = (
                    subject_signal_series.astype(float)
                    .reindex(range_returns.index)
                    .fillna(0.0)
                )
        subject_series.append(
            SubjectBacktestSeries(
                subject_id=subject_id,
                signal_series=signal_series,
                realized_return_series=range_returns.astype(float),
                historical_return_series=full_returns.astype(float),
                funding_cost_bps_series=(
                    None
                    if funding_cost_bps_series_by_subject is None
                    else funding_cost_bps_series_by_subject.get(subject_id)
                ),
                borrow_fee_bps_series=(
                    None
                    if borrow_fee_bps_series_by_subject is None
                    else borrow_fee_bps_series_by_subject.get(subject_id)
                ),
                roll_cost_bps_series=(
                    None
                    if roll_cost_bps_series_by_subject is None
                    else roll_cost_bps_series_by_subject.get(subject_id)
                ),
                contract_multiplier=(
                    None
                    if contract_multiplier_by_subject is None
                    else contract_multiplier_by_subject.get(subject_id)
                ),
            )
        )
    if not subject_series:
        return None
    prediction_diagnostics = build_prediction_diagnostics(
        signal_series_by_subject={
            item.subject_id: item.signal_series for item in subject_series
        },
        forward_return_series_by_subject={
            item.subject_id: item.realized_return_series for item in subject_series
        },
        group_by_subject=None,
    )
    return RangeBacktestDataset(
        label=date_range.label,
        predictive_corr=prediction_diagnostics.mean_signal_forward_corr,
        prediction_diagnostics=prediction_diagnostics,
        subject_series=tuple(subject_series),
    )


def evaluate_range_backtest_variants(
    *,
    subject_set_id: str | None,
    subject_set: SubjectSet | None,
    target_id: str,
    dataset: RangeBacktestDataset,
    portfolio_construction: PortfolioConstructionSpec,
    trading_environment: TradingEnvironment,
    top_k: int | None,
) -> RangeBacktestVariantResults:
    selected = _run_backtest_variant(
        subject_set_id=subject_set_id,
        subject_set=subject_set,
        target_id=target_id,
        subject_series=dataset.subject_series,
        dependence_series=dataset.dependence_series,
        portfolio_construction=portfolio_construction,
        trading_environment=trading_environment,
        top_k=top_k,
    )
    if _uses_expensive_sizing_optimizer(portfolio_construction):
        daily_rebalance = selected
    else:
        daily_rebalance = _run_backtest_variant(
            subject_set_id=subject_set_id,
            subject_set=subject_set,
            target_id=target_id,
            subject_series=dataset.subject_series,
            dependence_series=dataset.dependence_series,
            portfolio_construction=_portfolio_construction_variant(
                portfolio_construction,
                rebalance_interval_steps=1,
            ),
            trading_environment=trading_environment,
            top_k=top_k,
        )
    equal_weight = _run_backtest_variant(
        subject_set_id=subject_set_id,
        subject_set=subject_set,
        target_id=target_id,
        subject_series=dataset.subject_series,
        dependence_series=dataset.dependence_series,
        portfolio_construction=_portfolio_construction_variant(
            portfolio_construction,
            sizing_method="equal_weight",
        ),
        trading_environment=trading_environment,
        top_k=top_k,
    )
    equal_weight_daily = _run_backtest_variant(
        subject_set_id=subject_set_id,
        subject_set=subject_set,
        target_id=target_id,
        subject_series=dataset.subject_series,
        dependence_series=dataset.dependence_series,
        portfolio_construction=_portfolio_construction_variant(
            portfolio_construction,
            sizing_method="equal_weight",
            rebalance_interval_steps=1,
        ),
        trading_environment=trading_environment,
        top_k=top_k,
    )
    return RangeBacktestVariantResults(
        selected=selected,
        daily_rebalance=daily_rebalance,
        equal_weight=equal_weight,
        equal_weight_daily=equal_weight_daily,
    )


def _range_summary_from_variant_results(
    dataset: RangeBacktestDataset,
    variant_results: RangeBacktestVariantResults,
    *,
    portfolio_construction: PortfolioConstructionSpec,
    subject_set: SubjectSet | None,
) -> StrategyBacktestRangeSummary:
    selected = variant_results.selected
    concentration = _portfolio_concentration_from_backtest(
        selected,
        subject_set=subject_set,
        min_abs_weight=_CONCENTRATION_MIN_ABS_WEIGHT,
        top_intent_n=portfolio_construction.top_gross_share_cap_n,
    )
    construction_impact = _portfolio_construction_stage_impact_from_backtest(selected)
    execution_impact = _execution_trace_from_backtest(selected)
    cost_drag = _cost_drag_from_backtest(selected, subject_set=subject_set)
    signal_churn = _signal_churn_from_backtest(selected)
    optimizer_diagnostics = _optimizer_diagnostics_from_backtest(selected)
    return StrategyBacktestRangeSummary(
        label=dataset.label,
        predictive_corr=dataset.predictive_corr,
        prediction_hit_rate=dataset.prediction_diagnostics.mean_signal_hit_rate,
        prediction_long_short_spread=(
            dataset.prediction_diagnostics.mean_long_short_forward_spread
        ),
        prediction_long_bucket_return=dataset.prediction_diagnostics.long_bucket_return,
        prediction_short_bucket_return=dataset.prediction_diagnostics.short_bucket_return,
        prediction_coverage=dataset.prediction_diagnostics.coverage,
        prediction_positive_group_fraction=(
            dataset.prediction_diagnostics.positive_group_fraction
        ),
        portfolio_target_return_corr=_portfolio_target_return_corr(selected),
        decision_net_return=selected.net_return_total,
        decision_drawdown=selected.max_drawdown,
        decision_turnover=selected.mean_turnover,
        decision_gross_leverage_exposure=selected.mean_gross_leverage_exposure,
        decision_net_leverage_exposure=selected.mean_net_leverage_exposure,
        decision_long_leverage_exposure=selected.mean_long_leverage_exposure,
        decision_short_leverage_exposure=selected.mean_short_leverage_exposure,
        decision_gross_notional_exposure=selected.mean_gross_notional_exposure,
        decision_net_notional_exposure=selected.mean_net_notional_exposure,
        decision_long_notional_exposure=selected.mean_long_notional_exposure,
        decision_short_notional_exposure=selected.mean_short_notional_exposure,
        decision_traded_notional=selected.mean_traded_notional,
        decision_cost_notional=selected.cost_notional_total,
        decision_funding_cost_notional=selected.funding_cost_notional_total,
        decision_borrow_cost_notional=selected.borrow_cost_notional_total,
        decision_roll_cost_notional=selected.roll_cost_notional_total,
        optimizer_backend=str(optimizer_diagnostics["optimizer_backend"]),
        optimizer_status=str(optimizer_diagnostics["optimizer_status"]),
        optimizer_fallback_reason=str(
            optimizer_diagnostics["optimizer_fallback_reason"]
        ),
        target_vol_stage_mean_gross_delta=construction_impact[
            "target_vol_stage_mean_gross_delta"
        ],
        net_target_stage_mean_net_delta=construction_impact[
            "net_target_stage_mean_net_delta"
        ],
        top_k_stage_mean_active_count_delta=construction_impact[
            "top_k_stage_mean_active_count_delta"
        ],
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
        daily_rebalance_net_return=variant_results.daily_rebalance.net_return_total,
        daily_rebalance_drawdown=variant_results.daily_rebalance.max_drawdown,
        daily_rebalance_turnover=variant_results.daily_rebalance.mean_turnover,
        equal_weight_net_return=variant_results.equal_weight.net_return_total,
        equal_weight_drawdown=variant_results.equal_weight.max_drawdown,
        equal_weight_turnover=variant_results.equal_weight.mean_turnover,
        equal_weight_daily_net_return=variant_results.equal_weight_daily.net_return_total,
        equal_weight_daily_drawdown=variant_results.equal_weight_daily.max_drawdown,
        equal_weight_daily_turnover=variant_results.equal_weight_daily.mean_turnover,
        step_count=len(selected.steps),
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


def _portfolio_construction_stage_impact_from_backtest(
    result: DecisionBacktestResult,
) -> dict[str, float]:
    traces = [
        trace
        for step in result.steps
        for trace in getattr(step, "construction_trace", ())
    ]
    return {
        "target_vol_stage_mean_gross_delta": _mean(
            [
                item.gross_delta
                for item in traces
                if item.stage_name == "target_vol_cap"
            ]
        ),
        "net_target_stage_mean_net_delta": _mean(
            [
                item.net_delta
                for item in traces
                if item.stage_name == "net_exposure_target"
            ]
        ),
        "top_k_stage_mean_active_count_delta": _mean(
            [
                float(item.active_count_delta)
                for item in traces
                if item.stage_name == "top_k"
            ]
        ),
    }


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
    sizing_policy_cases: list[EvaluationFailureFinding] = []
    rebalance_policy_cases: list[EvaluationFailureFinding] = []
    portfolio_target_return_alignment_cases: list[EvaluationFailureFinding] = []
    concentration_cases: list[EvaluationFailureFinding] = []
    for item in sorted(
        range_summaries,
        key=lambda row: (
            row.decision_net_return,
            row.predictive_corr,
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
        if item.decision_net_return - item.equal_weight_net_return <= 0.0:
            sizing_policy_cases.append(
                EvaluationFailureFinding(
                    label=item.label,
                    severity=abs(
                        float(item.decision_net_return - item.equal_weight_net_return)
                    ),
                    metrics={
                        "selected_vs_equal_weight_decision_net_return_edge": round(
                            item.decision_net_return - item.equal_weight_net_return,
                            6,
                        ),
                        "decision_net_return": round(item.decision_net_return, 6),
                        "equal_weight_decision_net_return": round(
                            item.equal_weight_net_return,
                            6,
                        ),
                        "step_count": item.step_count,
                    },
                )
            )
        if item.decision_net_return - item.daily_rebalance_net_return <= 0.0:
            rebalance_policy_cases.append(
                EvaluationFailureFinding(
                    label=item.label,
                    severity=abs(
                        float(
                            item.decision_net_return - item.daily_rebalance_net_return
                        )
                    ),
                    metrics={
                        "selected_vs_daily_rebalance_net_return_edge": round(
                            item.decision_net_return - item.daily_rebalance_net_return,
                            6,
                        ),
                        "decision_net_return": round(item.decision_net_return, 6),
                        "daily_rebalance_net_return": round(
                            item.daily_rebalance_net_return,
                            6,
                        ),
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
            metric_group_name="sizing_policy_quality",
            source=source,
            findings=tuple(sizing_policy_cases),
        ),
        EvaluationFailureFindingGroup(
            metric_group_name="rebalance_policy_quality",
            source=source,
            findings=tuple(rebalance_policy_cases),
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
