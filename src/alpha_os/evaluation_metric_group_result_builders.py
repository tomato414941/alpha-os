from __future__ import annotations

from typing import Any

from .evaluation_result import EvaluationMetricGroupResult


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
