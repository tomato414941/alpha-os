from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cross_instrument_outcome import (
    CrossInstrumentOutcome,
    build_cross_instrument_outcome,
)


_METRIC_SCALAR_TYPES = (str, int, float, bool)


@dataclass(frozen=True)
class EvaluationMetricGroupResult:
    metric_group_name: str
    source: str
    metrics: dict[str, str | int | float | bool]

    def to_document(self) -> dict[str, Any]:
        return {
            "metric_group_name": self.metric_group_name,
            "source": self.source,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "EvaluationMetricGroupResult":
        if "dimension" in document:
            raise ValueError(
                "evaluation metric group result dimension field is no longer "
                "supported; use metric_group_name"
            )
        metric_group_name = document.get("metric_group_name")
        source = document.get("source", "native")
        metrics = document.get("metrics", {})
        if not isinstance(metric_group_name, str) or not metric_group_name:
            raise ValueError(
                "evaluation metric group result is missing metric_group_name"
            )
        if not isinstance(source, str) or not source:
            raise ValueError(
                "evaluation metric group result source is invalid: "
                f"{metric_group_name}"
            )
        if not isinstance(metrics, dict):
            raise ValueError(
                "evaluation metric group result metrics are invalid: "
                f"{metric_group_name}"
            )
        normalized_metrics: dict[str, str | int | float | bool] = {}
        for key, value in metrics.items():
            if not isinstance(value, _METRIC_SCALAR_TYPES):
                raise ValueError(
                    "evaluation metric group result metric must be scalar: "
                    f"{metric_group_name}.{key}"
                )
            normalized_metrics[str(key)] = value
        return cls(
            metric_group_name=metric_group_name,
            source=source,
            metrics=normalized_metrics,
        )


@dataclass(frozen=True)
class EvaluationFailureFinding:
    label: str
    severity: float
    metrics: dict[str, str | int | float | bool]

    def to_document(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "severity": self.severity,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "EvaluationFailureFinding":
        label = document.get("label")
        severity = document.get("severity", 0.0)
        metrics = document.get("metrics", {})
        if not isinstance(label, str) or not label:
            raise ValueError("evaluation failure finding is missing label")
        if not isinstance(severity, (int, float)):
            raise ValueError(
                f"evaluation failure finding severity is invalid: {label}"
            )
        if not isinstance(metrics, dict):
            raise ValueError(f"evaluation failure finding metrics are invalid: {label}")
        normalized_metrics: dict[str, str | int | float | bool] = {}
        for key, value in metrics.items():
            if not isinstance(value, _METRIC_SCALAR_TYPES):
                raise ValueError(
                    "evaluation failure finding metric must be scalar: "
                    f"{label}.{key}"
                )
            normalized_metrics[str(key)] = value
        return cls(
            label=label,
            severity=float(severity),
            metrics=normalized_metrics,
        )


@dataclass(frozen=True)
class EvaluationFailureFindingGroup:
    metric_group_name: str
    source: str
    findings: tuple[EvaluationFailureFinding, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "metric_group_name": self.metric_group_name,
            "source": self.source,
            "findings": [item.to_document() for item in self.findings],
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "EvaluationFailureFindingGroup":
        if "dimension" in document:
            raise ValueError(
                "evaluation failure finding group dimension field is no longer "
                "supported; use metric_group_name"
            )
        if "cases" in document:
            raise ValueError(
                "evaluation failure finding group cases field is no longer "
                "supported; use findings"
            )
        metric_group_name = document.get("metric_group_name")
        source = document.get("source", "native")
        findings = document.get("findings", [])
        if not isinstance(metric_group_name, str) or not metric_group_name:
            raise ValueError(
                "evaluation failure finding group is missing metric_group_name"
            )
        if not isinstance(source, str) or not source:
            raise ValueError(
                "evaluation failure finding group source is invalid: "
                f"{metric_group_name}"
            )
        if not isinstance(findings, list):
            raise ValueError(
                "evaluation failure finding group findings are invalid: "
                f"{metric_group_name}"
            )
        return cls(
            metric_group_name=metric_group_name,
            source=source,
            findings=tuple(
                EvaluationFailureFinding.from_document(item)
                for item in findings
                if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True, kw_only=True)
class EvaluationTaskResult:
    metric_group_results: tuple[EvaluationMetricGroupResult, ...] = ()
    failure_finding_groups: tuple[EvaluationFailureFindingGroup, ...] = ()
    cross_instrument_outcome: CrossInstrumentOutcome | None = None
    evaluation_task_id: str | None = None
    strategy_id: str | None = None

    def __post_init__(self) -> None:
        evaluation_task_id = self.evaluation_task_id
        strategy_id = self.strategy_id
        if not isinstance(evaluation_task_id, str) or not evaluation_task_id:
            raise ValueError("evaluation task result is missing evaluation_task_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise ValueError("evaluation task result is missing strategy_id")
        cross_instrument_outcome = self.cross_instrument_outcome
        if cross_instrument_outcome is None:
            cross_instrument_outcome = build_cross_instrument_outcome(
                metric_group_results=self.metric_group_results,
                failure_finding_groups=self.failure_finding_groups,
            )
        object.__setattr__(self, "evaluation_task_id", evaluation_task_id)
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "cross_instrument_outcome", cross_instrument_outcome)

    def to_document(self) -> dict[str, Any]:
        document = {
            "evaluation_task_id": self.evaluation_task_id,
            "strategy_id": self.strategy_id,
            "metric_group_results": [
                item.to_document() for item in self.metric_group_results
            ],
            "failure_finding_groups": [
                item.to_document() for item in self.failure_finding_groups
            ],
            "cross_instrument_outcome": self.cross_instrument_outcome.to_document(),
        }
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "EvaluationTaskResult":
        evaluation_task_id = document.get("evaluation_task_id")
        strategy_id = document.get("strategy_id")
        if "profiles" in document:
            raise ValueError(
                "evaluation task result profiles field is no longer supported; "
                "use metric_group_results"
            )
        if "dimension_results" in document:
            raise ValueError(
                "evaluation task result dimension_results field is no longer "
                "supported; use metric_group_results"
            )
        if "failure_profiles" in document:
            raise ValueError(
                "evaluation task result failure_profiles field is no longer "
                "supported; use failure_finding_groups"
            )
        if "failure_results" in document:
            raise ValueError(
                "evaluation task result failure_results field is no longer "
                "supported; use failure_finding_groups"
            )
        metric_group_results = document.get("metric_group_results", [])
        failure_finding_groups = document.get("failure_finding_groups", [])
        cross_instrument_outcome = document.get("cross_instrument_outcome")
        if not isinstance(metric_group_results, list):
            raise ValueError(
                "evaluation task result metric_group_results are invalid"
            )
        if not isinstance(failure_finding_groups, list):
            raise ValueError("evaluation task result failure_finding_groups are invalid")
        if cross_instrument_outcome is not None and not isinstance(cross_instrument_outcome, dict):
            raise ValueError("evaluation task result cross_instrument_outcome is invalid")
        return cls(
            evaluation_task_id=(
                None
                if evaluation_task_id is None
                else str(evaluation_task_id)
            ),
            strategy_id=None if strategy_id is None else str(strategy_id),
            metric_group_results=tuple(
                EvaluationMetricGroupResult.from_document(item)
                for item in metric_group_results
                if isinstance(item, dict)
            ),
            failure_finding_groups=tuple(
                EvaluationFailureFindingGroup.from_document(item)
                for item in failure_finding_groups
                if isinstance(item, dict)
            ),
            cross_instrument_outcome=(
                None
                if cross_instrument_outcome is None
                else CrossInstrumentOutcome.from_document(cross_instrument_outcome)
            ),
        )
