from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evaluation_metric_config import (
    DEFAULT_EVALUATION_AGGREGATION_KINDS,
    DEFAULT_METRIC_WINDOW,
    EVALUATION_METRIC_GROUP_NAMES,
    EvaluationMetricConfig,
)


@dataclass(frozen=True)
class EvaluationDateRange:
    label: str
    start_date: str
    end_date: str

    def to_document(self) -> dict[str, str]:
        return {
            "label": self.label,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "EvaluationDateRange":
        label = document.get("label")
        start_date = document.get("start_date")
        end_date = document.get("end_date")
        if not isinstance(label, str) or not label:
            raise ValueError("evaluation date range is missing label")
        if not isinstance(start_date, str) or not start_date:
            raise ValueError(f"evaluation date range {label} is missing start_date")
        if not isinstance(end_date, str) or not end_date:
            raise ValueError(f"evaluation date range {label} is missing end_date")
        return cls(label=label, start_date=start_date, end_date=end_date)


@dataclass(frozen=True)
class EvaluationFold:
    label: str
    execution_range: EvaluationDateRange
    evaluation_date_ranges: tuple[EvaluationDateRange, ...] = ()

    @property
    def resolved_evaluation_date_ranges(self) -> tuple[EvaluationDateRange, ...]:
        if self.evaluation_date_ranges:
            return self.evaluation_date_ranges
        return (self.execution_range,)

    def to_document(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "execution_range": self.execution_range.to_document(),
            "evaluation_date_ranges": [
                item.to_document() for item in self.evaluation_date_ranges
            ],
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "EvaluationFold":
        label = document.get("label")
        execution_range = document.get("execution_range")
        evaluation_date_ranges = document.get("evaluation_date_ranges", [])
        if not isinstance(label, str) or not label:
            raise ValueError("evaluation fold is missing label")
        if not isinstance(execution_range, dict):
            raise ValueError(f"evaluation fold is missing execution_range: {label}")
        if not isinstance(evaluation_date_ranges, list):
            raise ValueError(
                f"evaluation fold evaluation_date_ranges must be a list: {label}"
            )
        return cls(
            label=label,
            execution_range=EvaluationDateRange.from_document(execution_range),
            evaluation_date_ranges=tuple(
                EvaluationDateRange.from_document(item)
                for item in evaluation_date_ranges
                if isinstance(item, dict)
            ),
        )


def _validate_date_range(range_: EvaluationDateRange) -> None:
    if range_.start_date > range_.end_date:
        raise ValueError(
            f"evaluation date range has start_date after end_date: {range_.label}"
        )


def _validate_unique_labels(label_kind: str, labels: tuple[str, ...]) -> None:
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            f"evaluation spec contains duplicate {label_kind} labels: "
            + ", ".join(duplicates)
        )


def _validate_backtest_oos_like_folds(folds: tuple[EvaluationFold, ...]) -> None:
    _validate_unique_labels("fold", tuple(fold.label for fold in folds))
    for fold in folds:
        _validate_date_range(fold.execution_range)
        _validate_unique_labels(
            f"evaluation range for fold {fold.label}",
            tuple(item.label for item in fold.evaluation_date_ranges),
        )
        for item in fold.evaluation_date_ranges:
            _validate_date_range(item)


@dataclass(frozen=True)
class EvaluationSpec:
    execution_range: EvaluationDateRange
    metric_group_names: tuple[str, ...] = EVALUATION_METRIC_GROUP_NAMES
    evaluation_date_ranges: tuple[EvaluationDateRange, ...] = ()
    evaluation_folds: tuple[EvaluationFold, ...] = ()
    target_ids: tuple[str, ...] = ()
    metric_windows: tuple[int, ...] = (DEFAULT_METRIC_WINDOW,)
    aggregation_kinds: tuple[str, ...] = DEFAULT_EVALUATION_AGGREGATION_KINDS

    def __post_init__(self) -> None:
        self.metric_config
        if self.evaluation_folds and self.evaluation_date_ranges:
            raise ValueError(
                "evaluation spec cannot define both evaluation_folds and "
                "top-level evaluation_date_ranges"
            )
        _validate_date_range(self.execution_range)
        _validate_unique_labels(
            "top-level evaluation range",
            tuple(item.label for item in self.evaluation_date_ranges),
        )
        for item in self.evaluation_date_ranges:
            _validate_date_range(item)
        _validate_backtest_oos_like_folds(self.evaluation_folds)

    @property
    def metric_config(self) -> EvaluationMetricConfig:
        return EvaluationMetricConfig(
            metric_group_names=self.metric_group_names,
            metric_windows=self.metric_windows,
            aggregation_kinds=self.aggregation_kinds,
        )

    @property
    def resolved_evaluation_date_ranges(self) -> tuple[EvaluationDateRange, ...]:
        if self.evaluation_date_ranges:
            return self.evaluation_date_ranges
        return (self.execution_range,)

    @property
    def resolved_evaluation_folds(self) -> tuple[EvaluationFold, ...]:
        if self.evaluation_folds:
            return self.evaluation_folds
        return (
            EvaluationFold(
                label=self.execution_range.label,
                execution_range=self.execution_range,
                evaluation_date_ranges=self.resolved_evaluation_date_ranges,
            ),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "execution_range": self.execution_range.to_document(),
            "metric_group_names": list(self.metric_group_names),
            "evaluation_date_ranges": [
                item.to_document() for item in self.evaluation_date_ranges
            ],
            "evaluation_folds": [
                item.to_document() for item in self.evaluation_folds
            ],
            "target_ids": list(self.target_ids),
            "metric_windows": list(self.metric_windows),
            "aggregation_kinds": list(self.aggregation_kinds),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "EvaluationSpec":
        execution_range = document.get("execution_range")
        metric_config = EvaluationMetricConfig.from_document(document)
        evaluation_date_ranges = document.get("evaluation_date_ranges", [])
        evaluation_folds = document.get("evaluation_folds", [])
        target_ids = document.get("target_ids", [])
        if not isinstance(execution_range, dict):
            raise ValueError("evaluation spec is missing execution_range")
        if not isinstance(evaluation_date_ranges, list):
            raise ValueError("evaluation spec evaluation_date_ranges must be a list")
        if not isinstance(evaluation_folds, list):
            raise ValueError("evaluation spec evaluation_folds must be a list")
        if not isinstance(target_ids, list):
            raise ValueError("evaluation spec target_ids must be a list")
        return cls(
            execution_range=EvaluationDateRange.from_document(execution_range),
            metric_group_names=metric_config.metric_group_names,
            evaluation_date_ranges=tuple(
                EvaluationDateRange.from_document(item)
                for item in evaluation_date_ranges
                if isinstance(item, dict)
            ),
            evaluation_folds=tuple(
                EvaluationFold.from_document(item)
                for item in evaluation_folds
                if isinstance(item, dict)
            ),
            target_ids=tuple(str(item) for item in target_ids),
            metric_windows=metric_config.metric_windows,
            aggregation_kinds=metric_config.aggregation_kinds,
        )
