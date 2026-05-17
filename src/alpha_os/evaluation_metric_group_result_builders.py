from __future__ import annotations

from typing import Any

from .evaluation_result import EvaluationMetricGroupResult


def build_prediction_diagnostics_metric_group_result(
    *,
    source: str,
    range_summaries: list[Any],
    mean,
) -> EvaluationMetricGroupResult:
    return EvaluationMetricGroupResult(
        metric_group_name="prediction_diagnostics",
        source=source,
        metrics={
            "mean_signal_forward_corr": round(
                mean([item.predictive_corr for item in range_summaries]),
                6,
            ),
            "mean_signal_hit_rate": round(
                mean([item.prediction_hit_rate for item in range_summaries]),
                6,
            ),
            "mean_long_short_forward_spread": round(
                mean([item.prediction_long_short_spread for item in range_summaries]),
                6,
            ),
            "mean_long_bucket_return": round(
                mean([item.prediction_long_bucket_return for item in range_summaries]),
                6,
            ),
            "mean_short_bucket_return": round(
                mean([item.prediction_short_bucket_return for item in range_summaries]),
                6,
            ),
            "mean_prediction_coverage": round(
                mean([item.prediction_coverage for item in range_summaries]),
                6,
            ),
            "positive_group_fraction": round(
                mean(
                    [
                        item.prediction_positive_group_fraction
                        for item in range_summaries
                    ]
                ),
                6,
            ),
        },
    )


def build_portfolio_construction_trace_metric_group_result(
    *,
    source: str,
    range_summaries: list[Any],
    mean,
) -> EvaluationMetricGroupResult:
    return EvaluationMetricGroupResult(
        metric_group_name="portfolio_construction_trace",
        source=source,
        metrics={
            "risk_budget_stage_mean_gross_delta": round(
                mean(
                    [
                        item.risk_budget_stage_mean_gross_delta
                        for item in range_summaries
                    ]
                ),
                6,
            ),
            "target_vol_stage_mean_gross_delta": round(
                mean(
                    [item.target_vol_stage_mean_gross_delta for item in range_summaries]
                ),
                6,
            ),
            "net_target_stage_mean_net_delta": round(
                mean([item.net_target_stage_mean_net_delta for item in range_summaries]),
                6,
            ),
            "top_k_stage_mean_active_count_delta": round(
                mean(
                    [
                        item.top_k_stage_mean_active_count_delta
                        for item in range_summaries
                    ]
                ),
                6,
            ),
        },
    )
