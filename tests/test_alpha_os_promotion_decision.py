import pytest

from alpha_os.evaluation_report import EvaluationReport
from alpha_os.evaluation_result import (
    EvaluationMetricGroupResult,
    EvaluationTaskResult,
)
from alpha_os.promotion_decision import (
    PromotionDecision,
    PromotionRule,
    build_promotion_decision_id,
    decide_promotion,
)


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
    oos_contract_summary: dict[str, str] | None = None,
) -> EvaluationReport:
    return EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="eval:test",
        task_results=(candidate, baseline),
        created_at="2026-04-29T00:00:00Z",
        oos_contract_summary=(
            {
                "rigor_level": "backtest_oos",
                "enforcement": "strict",
                "date_parse": "pass",
                "range_non_overlap": "pass",
                "evaluation_after_execution": "pass",
                "strategy_checkpoint_required": "n/a",
            }
            if oos_contract_summary is None
            else oos_contract_summary
        ),
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
        created_at="2026-04-29T00:00:00Z",
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
        created_at="2026-04-29T00:00:00Z",
    )

    assert decision.status == "reject"
    assert decision.reasons == ("candidate mean decision net return edge is too low",)
    assert decision.metrics["mean_decision_net_return_edge"] == pytest.approx(-0.01)


def test_identical_candidate_and_baseline_does_not_promote():
    decision = decide_promotion(
        evaluation_report=_report(
            candidate=_task_result(
                "candidate",
                mean_net_return=0.08,
                worst_net_return=0.01,
                drawdown=0.05,
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
        created_at="2026-04-29T00:00:00Z",
    )

    assert decision.status == "reject"
    assert decision.reasons == ("candidate mean decision net return edge is too low",)
    assert decision.metrics["mean_decision_net_return_edge"] == pytest.approx(0.0)


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
        created_at="2026-04-29T00:00:00Z",
    )

    assert decision.status == "inconclusive"
    assert decision.reasons == (
        "task result candidate is missing numeric metric: "
        "robustness.worst_decision_net_return",
    )
    assert decision.metrics["candidate_worst_decision_net_return"] is None


def test_decide_promotion_is_inconclusive_without_strict_oos_evidence():
    decision = decide_promotion(
        evaluation_report=_report(
            candidate=_task_result(
                "candidate",
                mean_net_return=0.12,
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
            oos_contract_summary={},
        ),
        rule=PromotionRule(candidate_task_id="candidate", baseline_task_id="baseline"),
        created_at="2026-04-29T00:00:00Z",
    )

    assert decision.status == "inconclusive"
    assert decision.reasons == ("promotion requires strict OOS contract evidence",)


def test_decide_promotion_is_inconclusive_when_oos_contract_is_warn_only():
    decision = decide_promotion(
        evaluation_report=_report(
            candidate=_task_result(
                "candidate",
                mean_net_return=0.12,
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
            oos_contract_summary={
                "rigor_level": "diagnostic",
                "enforcement": "warn",
                "date_parse": "pass",
                "range_non_overlap": "pass",
                "evaluation_after_execution": "pass",
                "strategy_checkpoint_required": "n/a",
            },
        ),
        rule=PromotionRule(candidate_task_id="candidate", baseline_task_id="baseline"),
        created_at="2026-04-29T00:00:00Z",
    )

    assert decision.status == "inconclusive"
    assert decision.reasons == (
        "promotion requires OOS contract enforcement=strict",
        "promotion requires OOS rigor level",
    )


def test_decide_promotion_can_skip_strict_oos_requirement_for_diagnostics():
    decision = decide_promotion(
        evaluation_report=_report(
            candidate=_task_result(
                "candidate",
                mean_net_return=0.12,
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
            oos_contract_summary={},
        ),
        rule=PromotionRule(
            candidate_task_id="candidate",
            baseline_task_id="baseline",
            require_strict_oos=False,
        ),
        created_at="2026-04-29T00:00:00Z",
    )

    assert decision.status == "promote"


def test_decide_promotion_is_inconclusive_when_baseline_task_is_missing():
    report = EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="eval:test",
        task_results=(
            _task_result(
                "candidate",
                mean_net_return=0.12,
                worst_net_return=0.02,
                drawdown=0.04,
                turnover=0.10,
            ),
        ),
        created_at="2026-04-29T00:00:00Z",
        oos_contract_summary={
            "rigor_level": "backtest_oos",
            "enforcement": "strict",
            "date_parse": "pass",
            "range_non_overlap": "pass",
            "evaluation_after_execution": "pass",
            "strategy_checkpoint_required": "n/a",
        },
    )

    decision = decide_promotion(
        evaluation_report=report,
        rule=PromotionRule(candidate_task_id="candidate", baseline_task_id="baseline"),
        created_at="2026-04-29T00:00:00Z",
    )

    assert decision.status == "inconclusive"
    assert decision.reasons == ("evaluation report is missing task result: baseline",)


def test_same_strategy_candidate_and_baseline_do_not_promote():
    candidate = _task_result(
        "candidate",
        mean_net_return=0.12,
        worst_net_return=0.02,
        drawdown=0.04,
        turnover=0.10,
    )
    baseline = _task_result(
        "baseline",
        mean_net_return=0.08,
        worst_net_return=0.01,
        drawdown=0.05,
        turnover=0.10,
    )
    baseline = EvaluationTaskResult(
        evaluation_task_id=baseline.evaluation_task_id,
        strategy_id=candidate.strategy_id,
        metric_group_results=baseline.metric_group_results,
    )

    decision = decide_promotion(
        evaluation_report=_report(candidate=candidate, baseline=baseline),
        rule=PromotionRule(candidate_task_id="candidate", baseline_task_id="baseline"),
        created_at="2026-04-29T00:00:00Z",
    )

    assert decision.status == "reject"
    assert decision.reasons == ("candidate and baseline use the same strategy",)


def test_promotion_rule_roundtrips_document():
    rule = PromotionRule(
        candidate_task_id="candidate",
        baseline_task_id="baseline",
        min_mean_net_return_edge=0.02,
        max_worst_net_return_degradation=0.01,
        max_drawdown_degradation=0.03,
        max_turnover_ratio=1.5,
        require_strict_oos=True,
    )

    assert PromotionRule.from_document(rule.to_document()) == rule


def test_promotion_decision_roundtrips_document():
    decision = PromotionDecision(
        promotion_decision_id="report:test:promotion:candidate:vs:baseline",
        evaluation_report_id="report:test",
        candidate_task_id="candidate",
        baseline_task_id="baseline",
        rule=PromotionRule(candidate_task_id="candidate", baseline_task_id="baseline"),
        status="reject",
        reasons=("candidate mean decision net return edge is too low",),
        metrics={
            "candidate_task_id": "candidate",
            "baseline_task_id": "baseline",
            "mean_decision_net_return_edge": -0.01,
            "candidate_worst_decision_net_return": None,
        },
        created_at="2026-04-29T00:00:00Z",
    )

    assert PromotionDecision.from_document(decision.to_document()) == decision


def test_decide_promotion_sets_stable_decision_id_and_created_at():
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
        created_at="2026-04-29T00:00:00Z",
    )

    assert decision.promotion_decision_id == build_promotion_decision_id(
        evaluation_report_id="report:test",
        candidate_task_id="candidate",
        baseline_task_id="baseline",
    )
    assert decision.created_at == "2026-04-29T00:00:00Z"
