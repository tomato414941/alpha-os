import pytest

from alpha_os.evaluation_report import (
    EvaluationMetricGroupResult,
    EvaluationReport,
    EvaluationTaskResult,
)
from alpha_os.promotion_decision import PromotionRule, decide_promotion


def _task_result(
    task_id: str,
    *,
    mean_net_return: float | None,
    worst_net_return: float | None,
    drawdown: float | None,
    turnover: float | None,
) -> EvaluationTaskResult:
    decision_metrics = {}
    if mean_net_return is not None:
        decision_metrics["mean_decision_net_return"] = mean_net_return
    if drawdown is not None:
        decision_metrics["mean_decision_drawdown"] = drawdown
    if turnover is not None:
        decision_metrics["mean_decision_turnover"] = turnover

    robustness_metrics = {}
    if worst_net_return is not None:
        robustness_metrics["worst_decision_net_return"] = worst_net_return

    return EvaluationTaskResult(
        evaluation_task_id=task_id,
        strategy_id=f"strategy:{task_id}",
        metric_group_results=(
            EvaluationMetricGroupResult(
                metric_group_name="decision_quality",
                source="native_plan",
                metrics=decision_metrics,
            ),
            EvaluationMetricGroupResult(
                metric_group_name="robustness",
                source="native_plan",
                metrics=robustness_metrics,
            ),
        ),
    )


def _report(
    *,
    candidate: EvaluationTaskResult,
    baseline: EvaluationTaskResult,
) -> EvaluationReport:
    return EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="eval:test",
        task_results=(candidate, baseline),
        created_at="2026-04-29T00:00:00Z",
    )


def test_decide_promotion_promotes_candidate_that_beats_baseline():
    decision = decide_promotion(
        evaluation_report=_report(
            candidate=_task_result(
                "candidate",
                mean_net_return=0.12,
                worst_net_return=0.02,
                drawdown=0.04,
                turnover=0.12,
            ),
            baseline=_task_result(
                "baseline",
                mean_net_return=0.08,
                worst_net_return=0.01,
                drawdown=0.05,
                turnover=0.10,
            ),
        ),
        rule=PromotionRule(candidate_task_id="candidate", baseline_task_id="baseline"),
    )

    assert decision.status == "promote"
    assert decision.metrics["mean_decision_net_return_edge"] == pytest.approx(0.04)
    assert decision.metrics["mean_decision_turnover_ratio"] == pytest.approx(1.2)


def test_decide_promotion_rejects_candidate_with_lower_mean_net_return():
    decision = decide_promotion(
        evaluation_report=_report(
            candidate=_task_result(
                "candidate",
                mean_net_return=0.07,
                worst_net_return=0.02,
                drawdown=0.04,
                turnover=0.10,
            ),
            baseline=_task_result(
                "baseline",
                mean_net_return=0.08,
                worst_net_return=0.01,
                drawdown=0.05,
                turnover=0.10,
            ),
        ),
        rule=PromotionRule(candidate_task_id="candidate", baseline_task_id="baseline"),
    )

    assert decision.status == "reject"
    assert decision.reasons == ("candidate mean decision net return edge is too low",)
    assert decision.metrics["mean_decision_net_return_edge"] == pytest.approx(-0.01)


def test_decide_promotion_is_inconclusive_when_required_metric_is_missing():
    decision = decide_promotion(
        evaluation_report=_report(
            candidate=_task_result(
                "candidate",
                mean_net_return=0.12,
                worst_net_return=None,
                drawdown=0.04,
                turnover=0.10,
            ),
            baseline=_task_result(
                "baseline",
                mean_net_return=0.08,
                worst_net_return=0.01,
                drawdown=0.05,
                turnover=0.10,
            ),
        ),
        rule=PromotionRule(candidate_task_id="candidate", baseline_task_id="baseline"),
    )

    assert decision.status == "inconclusive"
    assert decision.reasons == (
        "task result candidate is missing numeric metric: "
        "robustness.worst_decision_net_return",
    )
    assert decision.metrics["candidate_worst_decision_net_return"] is None
