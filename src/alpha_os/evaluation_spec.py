from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal
import warnings

from .evaluation_metric_config import (
    DEFAULT_EVALUATION_AGGREGATION_KINDS,
    DEFAULT_METRIC_WINDOW,
    EvaluationMetricConfig,
)

EvaluationRigorLevel = Literal[
    "exploratory",
    "diagnostic",
    "backtest_oos",
    "fixed_state_oos",
    "operational",
]
OosContractEnforcement = Literal["off", "warn", "strict"]

EVALUATION_RIGOR_LEVELS = (
    "exploratory",
    "diagnostic",
    "backtest_oos",
    "fixed_state_oos",
    "operational",
)
OOS_CONTRACT_ENFORCEMENTS = ("off", "warn", "strict")


@dataclass(frozen=True)
class EvaluationOosContract:
    enforcement: OosContractEnforcement = "warn"
    require_non_overlapping_ranges: bool = True
    require_evaluation_after_execution: bool = True

    def __post_init__(self) -> None:
        if self.enforcement not in OOS_CONTRACT_ENFORCEMENTS:
            raise ValueError(f"unknown OOS contract enforcement: {self.enforcement}")

    def to_document(self) -> dict[str, Any]:
        return {
            "enforcement": self.enforcement,
            "require_non_overlapping_ranges": self.require_non_overlapping_ranges,
            "require_evaluation_after_execution": (self.require_evaluation_after_execution),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any] | None) -> "EvaluationOosContract":
        if document is None:
            return cls()
        if not isinstance(document, dict):
            raise ValueError("evaluation spec oos_contract must be an object")
        enforcement = document.get("enforcement", "warn")
        if not isinstance(enforcement, str):
            raise ValueError("evaluation spec oos_contract enforcement must be a string")
        return cls(
            enforcement=enforcement,
            require_non_overlapping_ranges=bool(
                document.get("require_non_overlapping_ranges", True)
            ),
            require_evaluation_after_execution=bool(
                document.get("require_evaluation_after_execution", True)
            ),
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


def _date_ranges_overlap(left: EvaluationDateRange, right: EvaluationDateRange) -> bool:
    return left.start_date <= right.end_date and right.start_date <= left.end_date


def _handle_oos_contract_violation(
    *,
    contract: EvaluationOosContract,
    message: str,
) -> None:
    if contract.enforcement == "strict":
        raise ValueError(message)
    if contract.enforcement == "warn":
        warnings.warn(message, UserWarning, stacklevel=3)


def _validate_oos_contract_for_ranges(
    *,
    contract: EvaluationOosContract,
    execution_range: EvaluationDateRange,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
) -> None:
    for evaluation_range in evaluation_date_ranges:
        if contract.require_non_overlapping_ranges and _date_ranges_overlap(
            execution_range, evaluation_range
        ):
            _handle_oos_contract_violation(
                contract=contract,
                message=(
                    "evaluation OOS contract violation: execution and evaluation "
                    f"ranges overlap: {execution_range.label} vs "
                    f"{evaluation_range.label}"
                ),
            )
        if (
            contract.require_evaluation_after_execution
            and evaluation_range.start_date <= execution_range.end_date
        ):
            _handle_oos_contract_violation(
                contract=contract,
                message=(
                    "evaluation OOS contract violation: evaluation range does not "
                    f"start after execution range: {execution_range.label} vs "
                    f"{evaluation_range.label}"
                ),
            )


def _validate_oos_contract(
    *,
    rigor_level: EvaluationRigorLevel,
    contract: EvaluationOosContract,
    execution_range: EvaluationDateRange,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
    evaluation_folds: tuple[EvaluationFold, ...],
) -> None:
    if contract.enforcement == "off":
        return
    if rigor_level == "exploratory" and contract.enforcement != "strict":
        return
    if evaluation_date_ranges:
        _validate_oos_contract_for_ranges(
            contract=contract,
            execution_range=execution_range,
            evaluation_date_ranges=evaluation_date_ranges,
        )
    for fold in evaluation_folds:
        if fold.evaluation_date_ranges:
            _validate_oos_contract_for_ranges(
                contract=contract,
                execution_range=fold.execution_range,
                evaluation_date_ranges=fold.evaluation_date_ranges,
            )


def _has_oos_contract_overlap(
    *,
    execution_range: EvaluationDateRange,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
) -> bool:
    return any(
        _date_ranges_overlap(execution_range, evaluation_range)
        for evaluation_range in evaluation_date_ranges
    )


def _has_oos_contract_evaluation_before_or_inside_execution(
    *,
    execution_range: EvaluationDateRange,
    evaluation_date_ranges: tuple[EvaluationDateRange, ...],
) -> bool:
    return any(
        evaluation_range.start_date <= execution_range.end_date
        for evaluation_range in evaluation_date_ranges
    )


def _contract_result_status(
    *,
    enabled: bool,
    violated: bool,
) -> str:
    if not enabled:
        return "n/a"
    return "warn" if violated else "pass"


def build_oos_contract_summary(spec: "EvaluationSpec") -> dict[str, str]:
    ranges = tuple(spec.resolved_evaluation_folds)
    has_overlap = any(
        _has_oos_contract_overlap(
            execution_range=fold.execution_range,
            evaluation_date_ranges=fold.resolved_evaluation_date_ranges,
        )
        for fold in ranges
    )
    has_evaluation_before_or_inside_execution = any(
        _has_oos_contract_evaluation_before_or_inside_execution(
            execution_range=fold.execution_range,
            evaluation_date_ranges=fold.resolved_evaluation_date_ranges,
        )
        for fold in ranges
    )
    return {
        "rigor_level": spec.rigor_level,
        "enforcement": spec.oos_contract.enforcement,
        "date_parse": "pass",
        "range_non_overlap": _contract_result_status(
            enabled=spec.oos_contract.require_non_overlapping_ranges,
            violated=has_overlap,
        ),
        "evaluation_after_execution": _contract_result_status(
            enabled=spec.oos_contract.require_evaluation_after_execution,
            violated=has_evaluation_before_or_inside_execution,
        ),
    }


def _validate_target_ids(target_ids: tuple[str, ...]) -> None:
    if any(not isinstance(item, str) or not item for item in target_ids):
        raise ValueError("evaluation spec target_ids must be non-empty strings")


@dataclass(frozen=True)
class EvaluationSpec:
    execution_range: EvaluationDateRange
    evaluation_date_ranges: tuple[EvaluationDateRange, ...] = ()
    evaluation_folds: tuple[EvaluationFold, ...] = ()
    target_ids: tuple[str, ...] = ()
    metric_windows: tuple[int, ...] = (DEFAULT_METRIC_WINDOW,)
    aggregation_kinds: tuple[str, ...] = DEFAULT_EVALUATION_AGGREGATION_KINDS
    rigor_level: EvaluationRigorLevel = "exploratory"
    oos_contract: EvaluationOosContract = EvaluationOosContract()

    def __post_init__(self) -> None:
        self.metric_config
        if self.rigor_level not in EVALUATION_RIGOR_LEVELS:
            raise ValueError(f"unknown evaluation rigor_level: {self.rigor_level}")
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
        _validate_target_ids(self.target_ids)
        _validate_oos_contract(
            rigor_level=self.rigor_level,
            contract=self.oos_contract,
            execution_range=self.execution_range,
            evaluation_date_ranges=self.evaluation_date_ranges,
            evaluation_folds=self.evaluation_folds,
        )

    @property
    def metric_config(self) -> EvaluationMetricConfig:
        return EvaluationMetricConfig(
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
            "evaluation_date_ranges": [item.to_document() for item in self.evaluation_date_ranges],
            "evaluation_folds": [item.to_document() for item in self.evaluation_folds],
            "target_ids": list(self.target_ids),
            "metric_windows": list(self.metric_windows),
            "aggregation_kinds": list(self.aggregation_kinds),
            "rigor_level": self.rigor_level,
            "oos_contract": self.oos_contract.to_document(),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "EvaluationSpec":
        execution_range = document.get("execution_range")
        metric_config = EvaluationMetricConfig.from_document(document)
        evaluation_date_ranges = document.get("evaluation_date_ranges", [])
        evaluation_folds = document.get("evaluation_folds", [])
        target_ids = document.get("target_ids", [])
        rigor_level = document.get("rigor_level", "exploratory")
        if not isinstance(execution_range, dict):
            raise ValueError("evaluation spec is missing execution_range")
        if not isinstance(evaluation_date_ranges, list):
            raise ValueError("evaluation spec evaluation_date_ranges must be a list")
        if not isinstance(evaluation_folds, list):
            raise ValueError("evaluation spec evaluation_folds must be a list")
        if not isinstance(target_ids, list):
            raise ValueError("evaluation spec target_ids must be a list")
        if not isinstance(rigor_level, str):
            raise ValueError("evaluation spec rigor_level must be a string")
        return cls(
            execution_range=EvaluationDateRange.from_document(execution_range),
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
            rigor_level=rigor_level,
            oos_contract=EvaluationOosContract.from_document(document.get("oos_contract")),
        )
