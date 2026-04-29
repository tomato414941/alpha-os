from __future__ import annotations

from dataclasses import dataclass

from .cross_instrument_contract import (
    CrossInstrumentReportContract,
    default_evaluation_report_cross_instrument_contract,
    default_validation_result_set_cross_instrument_contract,
)


@dataclass(frozen=True)
class EvaluationReportContractValidationResult:
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues


def _normalized_universe_policy_fields(
    universe_policy_fields: dict[str, str | None],
) -> dict[str, str]:
    return {
        key: value
        for key, value in universe_policy_fields.items()
        if value is not None
    }


def _append_contract_issue(
    issues: list[str],
    *,
    actual: CrossInstrumentReportContract,
    expected: CrossInstrumentReportContract,
    label: str,
) -> None:
    if actual != expected:
        issues.append(
            f"{label} contract does not match the canonical evaluation report contract"
        )


def _active_constraint_fields(constraint_stages: tuple[str, ...]) -> tuple[str, ...]:
    fields: list[str] = []
    for stage in constraint_stages:
        _, _, stage_fields = stage.partition(":")
        if not stage_fields:
            continue
        for field_name in stage_fields.split(","):
            normalized = field_name.strip()
            if normalized:
                fields.append(normalized)
    return tuple(fields)


def validate_evaluation_report_contract(
    *,
    validation_run,
    evaluation_report,
) -> EvaluationReportContractValidationResult:
    issues: list[str] = []

    validation_contract = getattr(
        validation_run,
        "cross_instrument_contract",
        default_validation_result_set_cross_instrument_contract(),
    )
    _append_contract_issue(
        issues,
        actual=validation_contract,
        expected=default_validation_result_set_cross_instrument_contract(),
        label="validation",
    )
    validation_result_set = getattr(validation_run, "validation_result_set", None)
    if validation_result_set is None:
        issues.append("validation result set is missing")
    else:
        if not validation_result_set.signal_summaries:
            issues.append("validation result set is missing signal-level comparisons")
        if not validation_result_set.meta_summaries:
            issues.append("validation result set is missing meta-aggregation comparisons")
        if not validation_result_set.decision_summaries:
            issues.append("validation result set is missing decision comparisons")
        validation_universe_policy_by_subject_set: dict[str, dict[str, str]] = {}
        for item in validation_result_set.decision_summaries:
            if item.subject_set_id is not None and not item.subject_set_contract_groups:
                issues.append(
                    "validation result set is missing subject-set contract groups "
                    f"for {item.subject_set_id}/{item.aggregation_kind}"
                )
            normalized_universe_policy = _normalized_universe_policy_fields(
                item.universe_policy_fields
            )
            if (
                item.subject_set_id is not None
                and "universe_policy" in item.subject_set_contract_groups
                and not normalized_universe_policy
            ):
                issues.append(
                    "validation result set is missing universe-policy fields "
                    f"for {item.subject_set_id}/{item.aggregation_kind}"
                )
            if item.subject_set_id is not None and normalized_universe_policy:
                validation_universe_policy_by_subject_set[item.subject_set_id] = (
                    normalized_universe_policy
                )
    if validation_result_set is None:
        validation_universe_policy_by_subject_set = {}

    report = evaluation_report.report if hasattr(evaluation_report, "report") else evaluation_report
    report_contract = getattr(
        report,
        "cross_instrument_contract",
        default_evaluation_report_cross_instrument_contract(),
    )
    _append_contract_issue(
        issues,
        actual=report_contract,
        expected=default_evaluation_report_cross_instrument_contract(),
        label="evaluation report",
    )
    if not report.task_results:
        issues.append("evaluation report is missing task results")
        return EvaluationReportContractValidationResult(issues=tuple(issues))

    report_metric_contracts = {
        (item.outcome_kind, item.metric_group_name): item.metric_fields
        for item in report_contract.metric_contracts
    }
    for task_result in report.task_results:
        if task_result.cross_instrument_outcome is None:
            issues.append(
                f"evaluation report task result {task_result.evaluation_task_id} is missing cross-instrument outcomes"
            )
            continue
        if not task_result.strategy_contract_fields:
            issues.append(
                f"evaluation report task result {task_result.evaluation_task_id} is missing strategy contract fields"
            )
        if task_result.subject_set_contract_groups and not task_result.subject_set_facts:
            issues.append(
                f"evaluation report task result {task_result.evaluation_task_id} is missing subject-set facts"
            )
        if not task_result.subject_set_contract_groups:
            issues.append(
                f"evaluation report task result {task_result.evaluation_task_id} is missing subject-set contract groups"
            )
        normalized_report_universe_policy = _normalized_universe_policy_fields(
            task_result.universe_policy_fields
        )
        if (
            "universe_policy" in task_result.subject_set_contract_groups
            and not normalized_report_universe_policy
        ):
            issues.append(
                f"evaluation report task result {task_result.evaluation_task_id} is missing universe-policy fields"
            )
        for key, value in normalized_report_universe_policy.items():
            strategy_value = task_result.strategy_contract_fields.get(key)
            if strategy_value != value:
                issues.append(
                    f"evaluation report task result {task_result.evaluation_task_id} strategy contract is missing universe-policy field {key}"
                )
        if not task_result.constraint_stages:
            issues.append(
                f"evaluation report task result {task_result.evaluation_task_id} is missing constraint stages"
            )
        for field_name in _active_constraint_fields(task_result.constraint_stages):
            if field_name not in task_result.strategy_contract_fields:
                issues.append(
                    f"evaluation report task result {task_result.evaluation_task_id} strategy contract is missing active constraint field {field_name}"
                )
        subject_set_id = task_result.strategy_contract_fields.get("subject_set")
        if (
            isinstance(subject_set_id, str)
            and subject_set_id in validation_universe_policy_by_subject_set
            and normalized_report_universe_policy
            != validation_universe_policy_by_subject_set[subject_set_id]
        ):
            issues.append(
                f"evaluation report task result {task_result.evaluation_task_id} universe-policy fields do not match validation result set for {subject_set_id}"
            )
        if not task_result.cross_instrument_outcome.metric_group_outcomes:
            issues.append(
                f"evaluation report task result {task_result.evaluation_task_id} is missing metric group outcomes"
            )
        if not task_result.cross_instrument_outcome.failure_finding_outcomes:
            issues.append(
                f"evaluation report task result {task_result.evaluation_task_id} is missing failure outcomes"
            )
        for outcome in task_result.cross_instrument_outcome.metric_group_outcomes:
            expected_fields = report_metric_contracts.get(
                ("metric_group_outcome", outcome.metric_group_name)
            )
            if expected_fields is None:
                issues.append(
                    f"evaluation report task result {task_result.evaluation_task_id} has no metric contract for metric group outcome {outcome.metric_group_name}"
                )
                continue
            if set(outcome.metrics.keys()) != set(expected_fields):
                issues.append(
                    f"evaluation report task result {task_result.evaluation_task_id} metric keys do not match the contract for metric group outcome {outcome.metric_group_name}"
                )
        for outcome in task_result.cross_instrument_outcome.failure_finding_outcomes:
            expected_fields = report_metric_contracts.get(
                ("failure_finding_outcome", outcome.metric_group_name)
            )
            if expected_fields is None:
                issues.append(
                    f"evaluation report task result {task_result.evaluation_task_id} has no metric contract for failure outcome {outcome.metric_group_name}"
                )
                continue
            if expected_fields != ("finding_count", "max_severity"):
                issues.append(
                    f"evaluation report task result {task_result.evaluation_task_id} failure contract is invalid for {outcome.metric_group_name}"
                )

    return EvaluationReportContractValidationResult(issues=tuple(issues))
