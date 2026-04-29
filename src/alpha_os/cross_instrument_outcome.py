from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_SCALAR_TYPES = (str, int, float, bool)


@dataclass(frozen=True)
class CrossInstrumentMetricGroupOutcome:
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
    def from_document(cls, document: dict[str, Any]) -> "CrossInstrumentMetricGroupOutcome":
        if "dimension" in document:
            raise ValueError(
                "cross-instrument metric group outcome dimension field is no longer "
                "supported; use metric_group_name"
            )
        metrics = document.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError("cross-instrument metric group outcome metrics are invalid")
        normalized_metrics: dict[str, str | int | float | bool] = {}
        for key, value in metrics.items():
            if not isinstance(value, _SCALAR_TYPES):
                raise ValueError(
                    f"cross-instrument metric group outcome metric must be scalar: {key}"
                )
            normalized_metrics[str(key)] = value
        return cls(
            metric_group_name=str(document["metric_group_name"]),
            source=str(document["source"]),
            metrics=normalized_metrics,
        )


@dataclass(frozen=True)
class CrossInstrumentFailureFindingOutcome:
    metric_group_name: str
    source: str
    finding_count: int
    max_severity: float

    def to_document(self) -> dict[str, Any]:
        return {
            "metric_group_name": self.metric_group_name,
            "source": self.source,
            "finding_count": self.finding_count,
            "max_severity": self.max_severity,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "CrossInstrumentFailureFindingOutcome":
        if "dimension" in document:
            raise ValueError(
                "cross-instrument failure finding outcome dimension field is no "
                "longer supported; use metric_group_name"
            )
        if "case_count" in document:
            raise ValueError(
                "cross-instrument failure finding outcome case_count field is no "
                "longer supported; use finding_count"
            )
        return cls(
            metric_group_name=str(document["metric_group_name"]),
            source=str(document["source"]),
            finding_count=int(document["finding_count"]),
            max_severity=float(document["max_severity"]),
        )


@dataclass(frozen=True)
class CrossInstrumentOutcome:
    metric_group_outcomes: tuple[CrossInstrumentMetricGroupOutcome, ...]
    failure_finding_outcomes: tuple[CrossInstrumentFailureFindingOutcome, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "metric_group_outcomes": [item.to_document() for item in self.metric_group_outcomes],
            "failure_finding_outcomes": [item.to_document() for item in self.failure_finding_outcomes],
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "CrossInstrumentOutcome":
        if "dimension_outcomes" in document:
            raise ValueError(
                "cross-instrument outcome dimension_outcomes field is no longer "
                "supported; use metric_group_outcomes"
            )
        if "failure_outcomes" in document:
            raise ValueError(
                "cross-instrument outcome failure_outcomes field is no longer "
                "supported; use failure_finding_outcomes"
            )
        metric_group_outcomes = document.get("metric_group_outcomes", [])
        failure_finding_outcomes = document.get("failure_finding_outcomes", [])
        if not isinstance(metric_group_outcomes, list):
            raise ValueError("cross-instrument outcome metric_group_outcomes are invalid")
        if not isinstance(failure_finding_outcomes, list):
            raise ValueError("cross-instrument outcome failure_finding_outcomes are invalid")
        return cls(
            metric_group_outcomes=tuple(
                CrossInstrumentMetricGroupOutcome.from_document(item)
                for item in metric_group_outcomes
                if isinstance(item, dict)
            ),
            failure_finding_outcomes=tuple(
                CrossInstrumentFailureFindingOutcome.from_document(item)
                for item in failure_finding_outcomes
                if isinstance(item, dict)
            ),
        )


def build_cross_instrument_outcome(
    *,
    metric_group_results: tuple[object, ...],
    failure_finding_groups: tuple[object, ...],
) -> CrossInstrumentOutcome:
    metric_group_outcomes = tuple(
        CrossInstrumentMetricGroupOutcome(
            metric_group_name=str(getattr(item, "metric_group_name")),
            source=str(getattr(item, "source")),
            metrics={
                str(key): value
                for key, value in dict(getattr(item, "metrics")).items()
                if isinstance(value, _SCALAR_TYPES)
            },
        )
        for item in metric_group_results
    )
    failure_finding_outcomes = tuple(
        CrossInstrumentFailureFindingOutcome(
            metric_group_name=str(getattr(item, "metric_group_name")),
            source=str(getattr(item, "source")),
            finding_count=len(tuple(getattr(item, "findings"))),
            max_severity=max(
                (
                    float(getattr(finding, "severity"))
                    for finding in tuple(getattr(item, "findings"))
                ),
                default=0.0,
            ),
        )
        for item in failure_finding_groups
    )
    return CrossInstrumentOutcome(
        metric_group_outcomes=metric_group_outcomes,
        failure_finding_outcomes=failure_finding_outcomes,
    )
