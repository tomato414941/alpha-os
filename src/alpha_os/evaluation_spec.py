from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


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

    def to_document(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "execution_range": self.execution_range.to_document(),
            "evaluation_date_ranges": [item.to_document() for item in self.evaluation_date_ranges],
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
            raise ValueError(f"evaluation fold evaluation_date_ranges must be a list: {label}")
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
    _parse_evaluation_date(range_.label, "start_date", range_.start_date)
    _parse_evaluation_date(range_.label, "end_date", range_.end_date)
    if range_.start_date > range_.end_date:
        raise ValueError(f"evaluation date range has start_date after end_date: {range_.label}")


def _parse_evaluation_date(label: str, field_name: str, value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"evaluation date range {label} has invalid {field_name}: {value}"
        ) from exc
    if parsed.isoformat() != value:
        raise ValueError(f"evaluation date range {label} has non-normalized {field_name}: {value}")
    return parsed


def _validate_unique_labels(label_kind: str, labels: tuple[str, ...]) -> None:
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            f"evaluation spec contains duplicate {label_kind} labels: " + ", ".join(duplicates)
        )


def _validate_evaluation_folds(folds: tuple[EvaluationFold, ...]) -> None:
    _validate_unique_labels("fold", tuple(fold.label for fold in folds))
    for fold in folds:
        _validate_date_range(fold.execution_range)
        _validate_unique_labels(
            f"evaluation range for fold {fold.label}",
            tuple(item.label for item in fold.evaluation_date_ranges),
        )
        for item in fold.evaluation_date_ranges:
            _validate_date_range(item)


def _date_ranges_overlap(left: EvaluationDateRange, right: EvaluationDateRange) -> bool:
    return left.start_date <= right.end_date and right.start_date <= left.end_date


def _validate_oos_ranges(
    *,
    execution_range: EvaluationDateRange,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
) -> None:
    for evaluation_range in evaluation_date_ranges:
        if _date_ranges_overlap(execution_range, evaluation_range):
            raise ValueError(
                "evaluation OOS violation: execution and evaluation ranges "
                f"overlap: {execution_range.label} vs {evaluation_range.label}"
            )
        if evaluation_range.start_date <= execution_range.end_date:
            raise ValueError(
                "evaluation OOS violation: evaluation range does not start after "
                f"execution range: {execution_range.label} vs "
                f"{evaluation_range.label}"
            )


def _validate_oos_ranges_for_spec(
    *,
    evaluation_folds: tuple[EvaluationFold, ...],
) -> None:
    for fold in evaluation_folds:
        if fold.evaluation_date_ranges:
            _validate_oos_ranges(
                execution_range=fold.execution_range,
                evaluation_date_ranges=fold.evaluation_date_ranges,
            )


@dataclass(frozen=True)
class EvaluationSpec:
    execution_range: EvaluationDateRange
    evaluation_folds: tuple[EvaluationFold, ...] = ()

    def __post_init__(self) -> None:
        _validate_date_range(self.execution_range)
        _validate_evaluation_folds(self.evaluation_folds)
        _validate_oos_ranges_for_spec(
            evaluation_folds=self.evaluation_folds,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "execution_range": self.execution_range.to_document(),
            "evaluation_folds": [item.to_document() for item in self.evaluation_folds],
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "EvaluationSpec":
        execution_range = document.get("execution_range")
        evaluation_folds = document.get("evaluation_folds", [])
        if not isinstance(execution_range, dict):
            raise ValueError("evaluation spec is missing execution_range")
        if not isinstance(evaluation_folds, list):
            raise ValueError("evaluation spec evaluation_folds must be a list")
        return cls(
            execution_range=EvaluationDateRange.from_document(execution_range),
            evaluation_folds=tuple(
                EvaluationFold.from_document(item)
                for item in evaluation_folds
                if isinstance(item, dict)
            ),
        )
