from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cross_instrument_contract import (
    CrossInstrumentReportContract,
    default_evaluation_report_cross_instrument_contract,
)
from .cross_instrument_outcome import (
    CrossInstrumentOutcome,
    build_cross_instrument_outcome,
)
from .evaluation_lane import normalize_evaluation_lane
from .strategy_sleeves import SleeveAttributionSummary


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
    evaluation_lane: str = "backtest_oos"
    construction_kind: str = "active_portfolio"
    metric_group_results: tuple[EvaluationMetricGroupResult, ...] = ()
    failure_finding_groups: tuple[EvaluationFailureFindingGroup, ...] = ()
    cross_instrument_outcome: CrossInstrumentOutcome | None = None
    strategy_contract_fields: dict[str, str | int | float | bool] = field(
        default_factory=dict
    )
    subject_set_facts: str | None = None
    subject_set_contract_groups: tuple[str, ...] = ()
    universe_policy_fields: dict[str, str | None] = field(default_factory=dict)
    constraint_stages: tuple[str, ...] = ()
    sleeve_attribution_summaries: tuple[SleeveAttributionSummary, ...] = ()
    artifact_refs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    evaluation_task_id: str | None = None
    strategy_id: str | None = None
    signal_discovery_id: str | None = None

    def __post_init__(self) -> None:
        evaluation_lane = normalize_evaluation_lane(self.evaluation_lane)
        evaluation_task_id = self.evaluation_task_id
        strategy_id = self.strategy_id
        if evaluation_task_id is None:
            values = self.artifact_refs.get("evaluation_task_ids", ())
            if values:
                evaluation_task_id = str(values[0])
        if strategy_id is None:
            values = self.artifact_refs.get("strategy_ids", ())
            if values:
                strategy_id = str(values[0])
        if evaluation_task_id is None:
            evaluation_task_id = self.signal_discovery_id
        if strategy_id is None:
            strategy_id = self.signal_discovery_id
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
        object.__setattr__(self, "evaluation_lane", evaluation_lane)
        object.__setattr__(self, "evaluation_task_id", evaluation_task_id)
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "cross_instrument_outcome", cross_instrument_outcome)

    def to_document(self) -> dict[str, Any]:
        document = {
            "evaluation_lane": self.evaluation_lane,
            "construction_kind": self.construction_kind,
            "evaluation_task_id": self.evaluation_task_id,
            "strategy_id": self.strategy_id,
            "metric_group_results": [
                item.to_document() for item in self.metric_group_results
            ],
            "failure_finding_groups": [item.to_document() for item in self.failure_finding_groups],
            "cross_instrument_outcome": self.cross_instrument_outcome.to_document(),
            "strategy_contract_fields": dict(self.strategy_contract_fields),
            "subject_set_facts": self.subject_set_facts,
            "subject_set_contract_groups": list(self.subject_set_contract_groups),
            "universe_policy_fields": dict(self.universe_policy_fields),
            "constraint_stages": list(self.constraint_stages),
            "sleeve_attribution_summaries": [
                item.to_document() for item in self.sleeve_attribution_summaries
            ],
            "artifact_refs": {
                key: list(value) for key, value in self.artifact_refs.items()
            },
        }
        if self.signal_discovery_id is not None:
            document["signal_discovery_id"] = self.signal_discovery_id
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "EvaluationTaskResult":
        evaluation_task_id = document.get("evaluation_task_id")
        strategy_id = document.get("strategy_id")
        signal_discovery_id = document.get("signal_discovery_id")
        evaluation_lane = document.get("evaluation_lane")
        construction_kind = document.get("construction_kind", "active_portfolio")
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
        if "subject_set_summary" in document:
            raise ValueError(
                "evaluation task result subject_set_summary field is no longer "
                "supported; use subject_set_facts"
            )
        metric_group_results = document.get("metric_group_results", [])
        failure_finding_groups = document.get("failure_finding_groups", [])
        cross_instrument_outcome = document.get("cross_instrument_outcome")
        strategy_contract_fields = document.get("strategy_contract_fields", {})
        subject_set_facts = document.get("subject_set_facts")
        subject_set_contract_groups = document.get("subject_set_contract_groups", [])
        universe_policy_fields = document.get("universe_policy_fields", {})
        constraint_stages = document.get("constraint_stages", [])
        sleeve_attribution_summaries = document.get("sleeve_attribution_summaries", [])
        artifact_refs = document.get("artifact_refs", {})
        if not isinstance(metric_group_results, list):
            raise ValueError(
                "evaluation task result metric_group_results are invalid"
            )
        if not isinstance(failure_finding_groups, list):
            raise ValueError("evaluation task result failure_finding_groups are invalid")
        if cross_instrument_outcome is not None and not isinstance(cross_instrument_outcome, dict):
            raise ValueError("evaluation task result cross_instrument_outcome is invalid")
        if not isinstance(strategy_contract_fields, dict):
            raise ValueError("evaluation task result strategy_contract_fields are invalid")
        if subject_set_facts is not None and not isinstance(subject_set_facts, str):
            raise ValueError("evaluation task result subject_set_facts is invalid")
        if not isinstance(subject_set_contract_groups, list):
            raise ValueError(
                "evaluation task result subject_set_contract_groups are invalid"
            )
        if not isinstance(universe_policy_fields, dict):
            raise ValueError("evaluation task result universe_policy_fields are invalid")
        if not isinstance(constraint_stages, list):
            raise ValueError("evaluation task result constraint_stages are invalid")
        if not isinstance(sleeve_attribution_summaries, list):
            raise ValueError(
                "evaluation task result sleeve_attribution_summaries are invalid"
            )
        if not isinstance(artifact_refs, dict):
            raise ValueError("evaluation task result artifact_refs are invalid")
        return cls(
            evaluation_task_id=(
                None
                if evaluation_task_id is None
                else str(evaluation_task_id)
            ),
            strategy_id=None if strategy_id is None else str(strategy_id),
            signal_discovery_id=(
                None if signal_discovery_id is None else str(signal_discovery_id)
            ),
            evaluation_lane=(
                None if evaluation_lane is None else str(evaluation_lane)
            ),
            construction_kind=str(construction_kind),
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
            strategy_contract_fields={
                str(key): value
                for key, value in strategy_contract_fields.items()
                if isinstance(value, _METRIC_SCALAR_TYPES)
            },
            subject_set_facts=subject_set_facts,
            subject_set_contract_groups=tuple(
                str(item) for item in subject_set_contract_groups
            ),
            universe_policy_fields={
                str(key): (
                    None if value is None else str(value)
                )
                for key, value in universe_policy_fields.items()
            },
            constraint_stages=tuple(str(item) for item in constraint_stages),
            sleeve_attribution_summaries=tuple(
                SleeveAttributionSummary.from_document(item)
                for item in sleeve_attribution_summaries
                if isinstance(item, dict)
            ),
            artifact_refs={
                str(key): tuple(str(item) for item in value)
                for key, value in artifact_refs.items()
                if isinstance(value, list)
            },
        )

@dataclass(frozen=True)
class EvaluationReport:
    evaluation_report_id: str
    evaluation_spec_id: str
    task_results: tuple[EvaluationTaskResult, ...]
    created_at: str
    evaluation_lane: str = "backtest_oos"
    oos_contract_summary: dict[str, str] = field(default_factory=dict)
    cross_instrument_contract: CrossInstrumentReportContract = field(
        default_factory=default_evaluation_report_cross_instrument_contract
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation_lane",
            normalize_evaluation_lane(self.evaluation_lane),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "evaluation_spec_id": self.evaluation_spec_id,
            "evaluation_lane": self.evaluation_lane,
            "oos_contract_summary": dict(self.oos_contract_summary),
            "task_results": [item.to_document() for item in self.task_results],
            "created_at": self.created_at,
            "cross_instrument_contract": self.cross_instrument_contract.to_document(),
        }

    @classmethod
    def from_document(
        cls,
        *,
        evaluation_report_id: str,
        document: dict[str, Any],
    ) -> "EvaluationReport":
        evaluation_spec_id = document.get("evaluation_spec_id")
        task_results = document.get("task_results", [])
        created_at = document.get("created_at")
        evaluation_lane = document.get("evaluation_lane")
        oos_contract_summary = document.get("oos_contract_summary", {})
        contract_document = document.get("cross_instrument_contract")
        if "summaries" in document:
            raise ValueError(
                "evaluation report summaries field is no longer supported; "
                "use task_results"
            )
        if "cross_instrument_criteria" in document:
            raise ValueError(
                "evaluation report cross_instrument_criteria field is no longer "
                "supported; use cross_instrument_contract"
            )
        if not isinstance(evaluation_spec_id, str) or not evaluation_spec_id:
            raise ValueError("evaluation report is missing evaluation_spec_id")
        if not isinstance(task_results, list):
            raise ValueError("evaluation report task_results are invalid")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("evaluation report is missing created_at")
        if not isinstance(oos_contract_summary, dict):
            raise ValueError("evaluation report oos_contract_summary is invalid")
        if contract_document is not None and not isinstance(contract_document, dict):
            raise ValueError("evaluation report cross_instrument_contract is invalid")
        return cls(
            evaluation_report_id=evaluation_report_id,
            evaluation_spec_id=evaluation_spec_id,
            task_results=tuple(
                EvaluationTaskResult.from_document(item)
                for item in task_results
                if isinstance(item, dict)
            ),
            created_at=created_at,
            evaluation_lane=None if evaluation_lane is None else str(evaluation_lane),
            oos_contract_summary={
                str(key): str(value) for key, value in oos_contract_summary.items()
            },
            cross_instrument_contract=(
                default_evaluation_report_cross_instrument_contract()
                if contract_document is None
                else CrossInstrumentReportContract.from_document(contract_document)
            ),
        )
