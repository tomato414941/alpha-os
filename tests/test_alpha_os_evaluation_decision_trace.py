from __future__ import annotations


def _build_trace_result():
    from alpha_os.decision_backtest import (
        DecisionBacktestResult,
        DecisionBacktestStep,
        DecisionBacktestSubjectStep,
    )
    subject_step = DecisionBacktestSubjectStep(
        subject_id="BTC_spot",
        signal_value=0.7,
        realized_return=0.02,
        target_weight=0.5,
        position_delta=0.5,
        target_notional=0.5,
        traded_notional=0.5,
        risk_scale=1.0,
        entry_allowed=True,
        gross_pnl_notional=0.0100,
        execution_cost_notional=0.0010,
        funding_cost_notional=0.0001,
        borrow_cost_notional=0.0,
        roll_cost_notional=0.0002,
        cost_notional=0.0013,
        net_pnl_notional=0.0087,
        net_return_contribution=0.0087,
        funding_cost_bps=2.0,
        borrow_fee_bps=3.0,
        roll_cost_bps=4.0,
        contract_multiplier=1.0,
        target_contracts=0.5,
        traded_contracts=0.5,
    )
    return DecisionBacktestResult(
        portfolio_id="portfolio:test",
        subject_set_id="core_crypto",
        target_id="residual_return_3d",
        subject_ids=("BTC_spot",),
        steps=(
            DecisionBacktestStep(
                date="2026-01-02",
                subject_steps=(subject_step,),
                gross_return=0.0100,
                gross_pnl_notional=0.0100,
                turnover=0.5,
                traded_notional=0.5,
                cost=0.0013,
                cost_notional=0.0013,
                net_return=0.0087,
                net_pnl_notional=0.0087,
                gross_leverage_exposure=0.5,
                net_leverage_exposure=0.5,
                long_leverage_exposure=0.5,
                short_leverage_exposure=0.0,
                gross_notional_exposure=0.5,
                net_notional_exposure=0.5,
                long_notional_exposure=0.5,
                short_notional_exposure=0.0,
                funding_cost_notional=0.0001,
                borrow_cost_notional=0.0,
                roll_cost_notional=0.0002,
                gross_equity=1.0100,
                net_equity=1.0087,
            ),
        ),
    )


def test_evaluation_run_result_lane_round_trips_and_defaults():
    from alpha_os.evaluation_run_result import EvaluationRunResult
    from alpha_os.evaluation_result import EvaluationResult

    default_lane = EvaluationRunResult.from_document(
        evaluation_run_result_id="run_result:legacy",
        document={
            "evaluation_spec_id": "evaluation_spec:test",
            "results": {
                "case:test": {
                    "strategy_id": "strategy:test",
                    "metric_group_results": [],
                    "failure_finding_groups": [],
                }
            },
            "created_at": "2026-04-21T00:00:00Z",
        },
    )
    assert default_lane.evaluation_lane == "backtest_oos"

    run_result = EvaluationRunResult(
        evaluation_run_result_id="run_result:test",
        evaluation_spec_id="evaluation_spec:test",
        evaluation_lane="diagnostic",
        oos_contract_summary={
            "rigor_level": "diagnostic",
            "enforcement": "warn",
            "date_parse": "pass",
            "range_non_overlap": "pass",
            "evaluation_after_execution": "pass",
        },
        results={
            "case:test": EvaluationResult(
                strategy_id="strategy:test",
            )
        },
        created_at="2026-04-21T00:00:00Z",
    )
    restored = EvaluationRunResult.from_document(
        evaluation_run_result_id="run_result:test",
        document=run_result.to_document(),
    )

    assert restored.evaluation_lane == "diagnostic"
    assert restored.oos_contract_summary["rigor_level"] == "diagnostic"
    assert restored.oos_contract_summary["enforcement"] == "warn"
