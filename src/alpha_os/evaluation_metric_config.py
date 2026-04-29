from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .scoring import DEFAULT_METRIC_WINDOW


DEFAULT_EVALUATION_AGGREGATION_KINDS = (
    "active_equal_mean",
    "corr_weighted_mean",
)


EVALUATION_METRIC_GROUP_NAMES = (
    "system_efficiency",
    "signal_discovery_quality",
    "signed_belief_quality",
    "prediction_diagnostics",
    "portfolio_target_return_alignment",
    "sizing_policy_quality",
    "rebalance_policy_quality",
    "decision_quality",
    "portfolio_risk_budget",
    "portfolio_construction_trace",
    "execution_trace",
    "cost_drag",
    "signal_churn",
    "portfolio_concentration",
    "robustness",
)


DECISION_EVALUATION_METRIC_GROUP_NAMES = (
    "signed_belief_quality",
    "prediction_diagnostics",
    "portfolio_target_return_alignment",
    "sizing_policy_quality",
    "rebalance_policy_quality",
    "decision_quality",
    "portfolio_risk_budget",
    "portfolio_construction_trace",
    "execution_trace",
    "cost_drag",
    "signal_churn",
    "portfolio_concentration",
    "robustness",
)


def requires_decision_evaluation(metric_group_names: tuple[str, ...]) -> bool:
    return any(item in metric_group_names for item in DECISION_EVALUATION_METRIC_GROUP_NAMES)


@dataclass(frozen=True)
class EvaluationMetricConfig:
    metric_group_names: tuple[str, ...] = EVALUATION_METRIC_GROUP_NAMES
    metric_windows: tuple[int, ...] = (DEFAULT_METRIC_WINDOW,)
    aggregation_kinds: tuple[str, ...] = DEFAULT_EVALUATION_AGGREGATION_KINDS

    def __post_init__(self) -> None:
        if not self.metric_group_names:
            raise ValueError("evaluation spec is missing metric_group_names")
        invalid_metric_group_names = [
            item
            for item in self.metric_group_names
            if item not in EVALUATION_METRIC_GROUP_NAMES
        ]
        if invalid_metric_group_names:
            joined = ", ".join(sorted(invalid_metric_group_names))
            raise ValueError(
                f"evaluation spec has unknown metric group names: {joined}"
            )
        if not self.metric_windows:
            raise ValueError("evaluation spec is missing metric_windows")
        if not self.aggregation_kinds:
            raise ValueError("evaluation spec is missing aggregation_kinds")

    def to_document(self) -> dict[str, Any]:
        return {
            "metric_group_names": list(self.metric_group_names),
            "metric_windows": list(self.metric_windows),
            "aggregation_kinds": list(self.aggregation_kinds),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "EvaluationMetricConfig":
        if "metric_group_names" in document and "dimensions" in document:
            raise ValueError(
                "evaluation spec cannot define both metric_group_names and "
                "legacy dimensions"
            )
        metric_group_names = document.get(
            "metric_group_names",
            document.get("dimensions", list(EVALUATION_METRIC_GROUP_NAMES)),
        )
        metric_windows = document.get("metric_windows", [DEFAULT_METRIC_WINDOW])
        aggregation_kinds = document.get(
            "aggregation_kinds",
            list(DEFAULT_EVALUATION_AGGREGATION_KINDS),
        )
        if not isinstance(metric_group_names, list) or not metric_group_names:
            raise ValueError("evaluation spec is missing metric_group_names")
        if not isinstance(metric_windows, list) or not metric_windows:
            raise ValueError("evaluation spec is missing metric_windows")
        if not isinstance(aggregation_kinds, list) or not aggregation_kinds:
            raise ValueError("evaluation spec is missing aggregation_kinds")
        return cls(
            metric_group_names=tuple(str(item) for item in metric_group_names),
            metric_windows=tuple(int(item) for item in metric_windows),
            aggregation_kinds=tuple(str(item) for item in aggregation_kinds),
        )
