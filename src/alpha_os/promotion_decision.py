from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


PromotionStatus = Literal["promote", "reject", "inconclusive"]


@dataclass(frozen=True)
class PromotionRule:
    candidate_task_id: str
    baseline_task_id: str
    min_mean_net_return_edge: float = 0.0
    max_worst_net_return_degradation: float = 0.0
    max_drawdown_degradation: float = 0.0
    max_turnover_ratio: float = 2.0

    def to_document(self) -> dict[str, str | float]:
        return {
            "candidate_task_id": self.candidate_task_id,
            "baseline_task_id": self.baseline_task_id,
            "min_mean_net_return_edge": self.min_mean_net_return_edge,
            "max_worst_net_return_degradation": self.max_worst_net_return_degradation,
            "max_drawdown_degradation": self.max_drawdown_degradation,
            "max_turnover_ratio": self.max_turnover_ratio,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "PromotionRule":
        candidate_task_id = document.get("candidate_task_id")
        baseline_task_id = document.get("baseline_task_id")
        if not isinstance(candidate_task_id, str) or not candidate_task_id:
            raise ValueError("promotion rule is missing candidate_task_id")
        if not isinstance(baseline_task_id, str) or not baseline_task_id:
            raise ValueError("promotion rule is missing baseline_task_id")
        return cls(
            candidate_task_id=candidate_task_id,
            baseline_task_id=baseline_task_id,
            min_mean_net_return_edge=float(
                document.get("min_mean_net_return_edge", 0.0)
            ),
            max_worst_net_return_degradation=float(
                document.get("max_worst_net_return_degradation", 0.0)
            ),
            max_drawdown_degradation=float(
                document.get("max_drawdown_degradation", 0.0)
            ),
            max_turnover_ratio=float(document.get("max_turnover_ratio", 2.0)),
        )


@dataclass(frozen=True)
class PromotionDecision:
    promotion_decision_id: str
    evaluation_report_id: str
    candidate_task_id: str
    baseline_task_id: str
    rule: PromotionRule
    status: PromotionStatus
    reasons: tuple[str, ...]
    metrics: dict[str, float | str | None]
    created_at: str

    def to_document(self) -> dict[str, object]:
        return {
            "promotion_decision_id": self.promotion_decision_id,
            "evaluation_report_id": self.evaluation_report_id,
            "candidate_task_id": self.candidate_task_id,
            "baseline_task_id": self.baseline_task_id,
            "rule": self.rule.to_document(),
            "status": self.status,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "created_at": self.created_at,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "PromotionDecision":
        promotion_decision_id = document.get("promotion_decision_id")
        evaluation_report_id = document.get("evaluation_report_id")
        candidate_task_id = document.get("candidate_task_id")
        baseline_task_id = document.get("baseline_task_id")
        rule = document.get("rule")
        status = document.get("status")
        reasons = document.get("reasons", [])
        metrics = document.get("metrics", {})
        created_at = document.get("created_at")
        if not isinstance(promotion_decision_id, str) or not promotion_decision_id:
            raise ValueError("promotion decision is missing promotion_decision_id")
        if not isinstance(evaluation_report_id, str) or not evaluation_report_id:
            raise ValueError("promotion decision is missing evaluation_report_id")
        if not isinstance(candidate_task_id, str) or not candidate_task_id:
            raise ValueError("promotion decision is missing candidate_task_id")
        if not isinstance(baseline_task_id, str) or not baseline_task_id:
            raise ValueError("promotion decision is missing baseline_task_id")
        if not isinstance(rule, dict):
            raise ValueError("promotion decision is missing rule")
        if status not in ("promote", "reject", "inconclusive"):
            raise ValueError("promotion decision status is invalid")
        if not isinstance(reasons, list) or any(
            not isinstance(item, str) for item in reasons
        ):
            raise ValueError("promotion decision reasons are invalid")
        if not isinstance(metrics, dict):
            raise ValueError("promotion decision metrics are invalid")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("promotion decision is missing created_at")
        return cls(
            promotion_decision_id=promotion_decision_id,
            evaluation_report_id=evaluation_report_id,
            candidate_task_id=candidate_task_id,
            baseline_task_id=baseline_task_id,
            rule=PromotionRule.from_document(rule),
            status=status,
            reasons=tuple(reasons),
            metrics={
                str(key): value
                for key, value in metrics.items()
                if value is None or isinstance(value, (int, float, str))
            },
            created_at=created_at,
        )


def build_promotion_decision_id(
    *,
    evaluation_report_id: str,
    candidate_task_id: str,
    baseline_task_id: str,
) -> str:
    return (
        f"{evaluation_report_id}:promotion:"
        f"{candidate_task_id}:vs:{baseline_task_id}"
    )


def decide_promotion(
    *,
    evaluation_report,
    rule: PromotionRule,
    created_at: str,
) -> PromotionDecision:
    metrics: dict[str, float | str | None] = {
        "candidate_task_id": rule.candidate_task_id,
        "baseline_task_id": rule.baseline_task_id,
    }
    missing_reasons: list[str] = []

    candidate_metrics = _task_metrics(
        evaluation_report=evaluation_report,
        task_id=rule.candidate_task_id,
        missing_reasons=missing_reasons,
    )
    baseline_metrics = _task_metrics(
        evaluation_report=evaluation_report,
        task_id=rule.baseline_task_id,
        missing_reasons=missing_reasons,
    )

    for side, side_metrics in (
        ("candidate", candidate_metrics),
        ("baseline", baseline_metrics),
    ):
        for key, value in side_metrics.items():
            metrics[f"{side}_{key}"] = value

    if missing_reasons:
        return PromotionDecision(
            promotion_decision_id=build_promotion_decision_id(
                evaluation_report_id=evaluation_report.evaluation_report_id,
                candidate_task_id=rule.candidate_task_id,
                baseline_task_id=rule.baseline_task_id,
            ),
            evaluation_report_id=evaluation_report.evaluation_report_id,
            candidate_task_id=rule.candidate_task_id,
            baseline_task_id=rule.baseline_task_id,
            rule=rule,
            status="inconclusive",
            reasons=tuple(missing_reasons),
            metrics=metrics,
            created_at=created_at,
        )

    mean_net_return_edge = (
        candidate_metrics["mean_decision_net_return"]
        - baseline_metrics["mean_decision_net_return"]
    )
    worst_net_return_edge = (
        candidate_metrics["worst_decision_net_return"]
        - baseline_metrics["worst_decision_net_return"]
    )
    drawdown_degradation = (
        candidate_metrics["mean_decision_drawdown"]
        - baseline_metrics["mean_decision_drawdown"]
    )
    turnover_ratio = _turnover_ratio(
        candidate_metrics["mean_decision_turnover"],
        baseline_metrics["mean_decision_turnover"],
    )
    metrics.update(
        {
            "mean_decision_net_return_edge": mean_net_return_edge,
            "worst_decision_net_return_edge": worst_net_return_edge,
            "mean_decision_drawdown_degradation": drawdown_degradation,
            "mean_decision_turnover_ratio": turnover_ratio,
        }
    )

    reject_reasons = []
    if mean_net_return_edge <= rule.min_mean_net_return_edge:
        reject_reasons.append("candidate mean decision net return edge is too low")
    if worst_net_return_edge < -rule.max_worst_net_return_degradation:
        reject_reasons.append("candidate worst decision net return degradation is too high")
    if drawdown_degradation > rule.max_drawdown_degradation:
        reject_reasons.append("candidate mean decision drawdown degradation is too high")
    if turnover_ratio is None or turnover_ratio > rule.max_turnover_ratio:
        reject_reasons.append("candidate mean decision turnover ratio is too high")

    if reject_reasons:
        return PromotionDecision(
            promotion_decision_id=build_promotion_decision_id(
                evaluation_report_id=evaluation_report.evaluation_report_id,
                candidate_task_id=rule.candidate_task_id,
                baseline_task_id=rule.baseline_task_id,
            ),
            evaluation_report_id=evaluation_report.evaluation_report_id,
            candidate_task_id=rule.candidate_task_id,
            baseline_task_id=rule.baseline_task_id,
            rule=rule,
            status="reject",
            reasons=tuple(reject_reasons),
            metrics=metrics,
            created_at=created_at,
        )

    return PromotionDecision(
        promotion_decision_id=build_promotion_decision_id(
            evaluation_report_id=evaluation_report.evaluation_report_id,
            candidate_task_id=rule.candidate_task_id,
            baseline_task_id=rule.baseline_task_id,
        ),
        evaluation_report_id=evaluation_report.evaluation_report_id,
        candidate_task_id=rule.candidate_task_id,
        baseline_task_id=rule.baseline_task_id,
        rule=rule,
        status="promote",
        reasons=("candidate satisfies promotion rule",),
        metrics=metrics,
        created_at=created_at,
    )


def _task_metrics(
    *,
    evaluation_report,
    task_id: str,
    missing_reasons: list[str],
) -> dict[str, float | None]:
    task_result = next(
        (
            item
            for item in evaluation_report.task_results
            if item.evaluation_task_id == task_id
        ),
        None,
    )
    if task_result is None:
        missing_reasons.append(f"evaluation report is missing task result: {task_id}")
        return {}

    return {
        "mean_decision_net_return": _metric(
            task_result=task_result,
            task_id=task_id,
            metric_group_name="decision_quality",
            metric_name="mean_decision_net_return",
            missing_reasons=missing_reasons,
        ),
        "worst_decision_net_return": _metric(
            task_result=task_result,
            task_id=task_id,
            metric_group_name="robustness",
            metric_name="worst_decision_net_return",
            missing_reasons=missing_reasons,
        ),
        "mean_decision_drawdown": _metric(
            task_result=task_result,
            task_id=task_id,
            metric_group_name="decision_quality",
            metric_name="mean_decision_drawdown",
            missing_reasons=missing_reasons,
        ),
        "mean_decision_turnover": _metric(
            task_result=task_result,
            task_id=task_id,
            metric_group_name="decision_quality",
            metric_name="mean_decision_turnover",
            missing_reasons=missing_reasons,
        ),
    }


def _metric(
    *,
    task_result,
    task_id: str,
    metric_group_name: str,
    metric_name: str,
    missing_reasons: list[str],
) -> float | None:
    metric_group = next(
        (
            item
            for item in task_result.metric_group_results
            if item.metric_group_name == metric_group_name
        ),
        None,
    )
    if metric_group is None:
        missing_reasons.append(
            f"task result {task_id} is missing metric group: {metric_group_name}"
        )
        return None
    value = metric_group.metrics.get(metric_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        missing_reasons.append(
            f"task result {task_id} is missing numeric metric: "
            f"{metric_group_name}.{metric_name}"
        )
        return None
    return float(value)


def _turnover_ratio(candidate_turnover: float, baseline_turnover: float) -> float | None:
    if baseline_turnover == 0.0:
        if candidate_turnover == 0.0:
            return 1.0
        return None
    return candidate_turnover / baseline_turnover
