from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_METRIC_WINDOW = 20
DEFAULT_EVALUATION_AGGREGATION_KINDS = (
    "active_equal_mean",
    "corr_weighted_mean",
)
EVALUATION_AGGREGATION_KINDS = DEFAULT_EVALUATION_AGGREGATION_KINDS


@dataclass(frozen=True)
class EvaluationMetricConfig:
    metric_windows: tuple[int, ...] = (DEFAULT_METRIC_WINDOW,)
    aggregation_kinds: tuple[str, ...] = DEFAULT_EVALUATION_AGGREGATION_KINDS

    def __post_init__(self) -> None:
        if not self.metric_windows:
            raise ValueError("evaluation spec is missing metric_windows")
        invalid_metric_windows = [
            item
            for item in self.metric_windows
            if not isinstance(item, int) or isinstance(item, bool) or item <= 0
        ]
        if invalid_metric_windows:
            joined = ", ".join(str(item) for item in invalid_metric_windows)
            raise ValueError(
                f"evaluation spec metric_windows must be positive integers: {joined}"
            )
        if not self.aggregation_kinds:
            raise ValueError("evaluation spec is missing aggregation_kinds")
        invalid_aggregation_kinds = [
            item
            for item in self.aggregation_kinds
            if item not in EVALUATION_AGGREGATION_KINDS
        ]
        if invalid_aggregation_kinds:
            joined = ", ".join(sorted(invalid_aggregation_kinds))
            raise ValueError(
                f"evaluation spec has unknown aggregation kinds: {joined}"
            )

    def to_document(self) -> dict[str, Any]:
        return {
            "metric_windows": list(self.metric_windows),
            "aggregation_kinds": list(self.aggregation_kinds),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "EvaluationMetricConfig":
        metric_windows = document.get("metric_windows", [DEFAULT_METRIC_WINDOW])
        aggregation_kinds = document.get(
            "aggregation_kinds",
            list(DEFAULT_EVALUATION_AGGREGATION_KINDS),
        )
        if not isinstance(metric_windows, list) or not metric_windows:
            raise ValueError("evaluation spec is missing metric_windows")
        if not isinstance(aggregation_kinds, list) or not aggregation_kinds:
            raise ValueError("evaluation spec is missing aggregation_kinds")
        try:
            normalized_metric_windows = tuple(int(item) for item in metric_windows)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "evaluation spec metric_windows must be positive integers"
            ) from exc
        return cls(
            metric_windows=normalized_metric_windows,
            aggregation_kinds=tuple(str(item) for item in aggregation_kinds),
        )
