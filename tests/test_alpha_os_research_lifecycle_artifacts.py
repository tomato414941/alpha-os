from alpha_os.baseline_state import baseline_from_promotion_decision
from alpha_os.evaluation_report import EvaluationReport
from alpha_os.evaluation_result import (
    EvaluationMetricGroupResult,
    EvaluationTaskResult,
)
from alpha_os.promotion_decision import PromotionRule, decide_promotion


def _task_result(
    task_id: str,
    *,
    mean_net_return: float,
    worst_net_return: float,
    drawdown: float,
    turnover: float,
) -> EvaluationTaskResult:
    return EvaluationTaskResult(
        evaluation_task_id=task_id,
        strategy_id=f"strategy:{task_id}",
        metric_group_results=(
            EvaluationMetricGroupResult(
                metric_group_name="decision_quality",
                source="native_plan",
                metrics={
                    "mean_decision_net_return": mean_net_return,
                    "mean_decision_drawdown": drawdown,
                    "mean_decision_turnover": turnover,
                },
            ),
            EvaluationMetricGroupResult(
                metric_group_name="robustness",
                source="native_plan",
                metrics={"worst_decision_net_return": worst_net_return},
            ),
        ),
    )


def test_promoted_evaluation_report_can_create_baseline_state():
    report = EvaluationReport(
        evaluation_report_id="report:lifecycle",
        evaluation_spec_id="eval:lifecycle",
        task_results=(
            _task_result(
                "candidate",
                mean_net_return=0.12,
                worst_net_return=0.03,
                drawdown=0.04,
                turnover=0.12,
            ),
            _task_result(
                "baseline",
                mean_net_return=0.08,
                worst_net_return=0.02,
                drawdown=0.05,
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
    baseline = baseline_from_promotion_decision(
        baseline_id="baseline:lifecycle:v1",
        strategy_id="strategy:candidate",
        promotion_decision=decision,
        active_from="2026-04-29T00:00:00Z",
    )

    assert decision.status == "promote"
    assert baseline.status == "active"
    assert baseline.source_promotion_decision_id == decision.promotion_decision_id
