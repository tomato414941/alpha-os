import pytest

from alpha_os.baseline_state import BaselineState, baseline_from_promotion_decision
from alpha_os.promotion_decision import PromotionDecision, PromotionRule


def _promotion_decision(status="promote") -> PromotionDecision:
    return PromotionDecision(
        promotion_decision_id="report:test:promotion:candidate:vs:baseline",
        evaluation_report_id="report:test",
        candidate_task_id="candidate",
        baseline_task_id="baseline",
        rule=PromotionRule(candidate_task_id="candidate", baseline_task_id="baseline"),
        status=status,
        reasons=("candidate satisfies promotion rule",),
        metrics={},
        created_at="2026-04-29T00:00:00Z",
    )


def test_baseline_state_roundtrips_document():
    baseline = BaselineState(
        baseline_id="baseline:macro:v1",
        strategy_id="strategy:candidate",
        source_promotion_decision_id="report:test:promotion:candidate:vs:baseline",
        active_from="2026-04-29T00:00:00Z",
        status="active",
    )

    assert BaselineState.from_document(baseline.to_document()) == baseline


def test_baseline_from_promoted_decision_creates_active_baseline():
    baseline = baseline_from_promotion_decision(
        baseline_id="baseline:macro:v1",
        strategy_id="strategy:candidate",
        promotion_decision=_promotion_decision("promote"),
        active_from="2026-04-29T00:00:00Z",
    )

    assert baseline.status == "active"
    assert baseline.strategy_id == "strategy:candidate"
    assert (
        baseline.source_promotion_decision_id
        == "report:test:promotion:candidate:vs:baseline"
    )


def test_baseline_from_rejected_decision_is_rejected():
    with pytest.raises(
        ValueError,
        match="baseline can only be created from promoted decision",
    ):
        baseline_from_promotion_decision(
            baseline_id="baseline:macro:v1",
            strategy_id="strategy:candidate",
            promotion_decision=_promotion_decision("reject"),
            active_from="2026-04-29T00:00:00Z",
        )
