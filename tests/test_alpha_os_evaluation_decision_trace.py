from __future__ import annotations

from types import SimpleNamespace

import pytest


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


def _build_tail_risk_trace_result():
    from alpha_os.decision_backtest import (
        DecisionBacktestResult,
        DecisionBacktestStep,
        DecisionBacktestSubjectStep,
    )

    return DecisionBacktestResult(
        portfolio_id="portfolio:test",
        subject_set_id="macro",
        target_id="residual_return_3d",
        subject_ids=("EQ_future", "TY_future"),
        steps=(
            DecisionBacktestStep(
                date="2017-01-03",
                subject_steps=(
                    DecisionBacktestSubjectStep(
                        subject_id="EQ_future",
                        signal_value=0.8,
                        realized_return=0.05,
                        target_weight=0.6,
                        position_delta=0.6,
                        target_notional=0.6,
                        traded_notional=0.6,
                        risk_scale=0.9,
                        entry_allowed=True,
                        gross_pnl_notional=0.0300,
                        execution_cost_notional=0.0010,
                        cost_notional=0.0010,
                        net_pnl_notional=0.0290,
                        net_return_contribution=0.0290,
                    ),
                    DecisionBacktestSubjectStep(
                        subject_id="TY_future",
                        signal_value=-0.4,
                        realized_return=0.02,
                        target_weight=-0.4,
                        position_delta=-0.4,
                        target_notional=-0.4,
                        traded_notional=0.4,
                        risk_scale=1.0,
                        entry_allowed=True,
                        gross_pnl_notional=-0.0080,
                        execution_cost_notional=0.0005,
                        cost_notional=0.0005,
                        net_pnl_notional=-0.0085,
                        net_return_contribution=-0.0085,
                    ),
                ),
                gross_return=0.0220,
                gross_pnl_notional=0.0220,
                turnover=1.0,
                traded_notional=1.0,
                cost=0.0015,
                cost_notional=0.0015,
                net_return=0.0205,
                net_pnl_notional=0.0205,
                gross_leverage_exposure=1.0,
                net_leverage_exposure=0.2,
                long_leverage_exposure=0.6,
                short_leverage_exposure=0.4,
                gross_notional_exposure=1.0,
                net_notional_exposure=0.2,
                long_notional_exposure=0.6,
                short_notional_exposure=0.4,
                funding_cost_notional=0.0,
                borrow_cost_notional=0.0,
                roll_cost_notional=0.0,
                gross_equity=1.0220,
                net_equity=1.0205,
            ),
            DecisionBacktestStep(
                date="2017-01-04",
                subject_steps=(
                    DecisionBacktestSubjectStep(
                        subject_id="EQ_future",
                        signal_value=0.7,
                        realized_return=-0.10,
                        target_weight=0.6,
                        position_delta=0.0,
                        target_notional=0.6,
                        traded_notional=0.0,
                        risk_scale=0.8,
                        entry_allowed=True,
                        gross_pnl_notional=-0.0600,
                        execution_cost_notional=0.0010,
                        cost_notional=0.0010,
                        net_pnl_notional=-0.0610,
                        net_return_contribution=-0.0610,
                    ),
                    DecisionBacktestSubjectStep(
                        subject_id="TY_future",
                        signal_value=-0.3,
                        realized_return=-0.05,
                        target_weight=-0.4,
                        position_delta=0.0,
                        target_notional=-0.4,
                        traded_notional=0.0,
                        risk_scale=1.0,
                        entry_allowed=True,
                        gross_pnl_notional=0.0200,
                        execution_cost_notional=0.0005,
                        cost_notional=0.0005,
                        net_pnl_notional=0.0195,
                        net_return_contribution=0.0195,
                    ),
                ),
                gross_return=-0.0400,
                gross_pnl_notional=-0.0400,
                turnover=0.0,
                traded_notional=0.0,
                cost=0.0015,
                cost_notional=0.0015,
                net_return=-0.0415,
                net_pnl_notional=-0.0415,
                gross_leverage_exposure=1.0,
                net_leverage_exposure=0.2,
                long_leverage_exposure=0.6,
                short_leverage_exposure=0.4,
                gross_notional_exposure=1.0,
                net_notional_exposure=0.2,
                long_notional_exposure=0.6,
                short_notional_exposure=0.4,
                funding_cost_notional=0.0,
                borrow_cost_notional=0.0,
                roll_cost_notional=0.0,
                gross_equity=0.9811,
                net_equity=0.9782,
            ),
        ),
    )


def _persist_report(store, *, report_id: str = "report:test") -> None:
    from alpha_os.evaluation_report import EvaluationReport

    store.upsert_evaluation_report(
        report=EvaluationReport(
            evaluation_report_id=report_id,
            evaluation_spec_id="evaluation_spec:test",
            task_results=(),
            created_at="2026-04-21T00:00:00Z",
        )
    )


def test_evaluation_report_lane_round_trips_and_defaults():
    from alpha_os.evaluation_report import EvaluationTaskResult, EvaluationReport

    legacy = EvaluationReport.from_document(
        evaluation_report_id="report:legacy",
        document={
            "evaluation_spec_id": "evaluation_spec:test",
            "task_results": [],
            "created_at": "2026-04-21T00:00:00Z",
        },
    )
    assert legacy.evaluation_lane == "backtest_oos"

    with pytest.raises(ValueError, match="summaries field is no longer supported"):
        EvaluationReport.from_document(
            evaluation_report_id="report:legacy-summaries",
            document={
                "evaluation_spec_id": "evaluation_spec:test",
                "summaries": [],
                "created_at": "2026-04-21T00:00:00Z",
            },
        )

    with pytest.raises(
        ValueError,
        match="cross_instrument_criteria field is no longer supported",
    ):
        EvaluationReport.from_document(
            evaluation_report_id="report:legacy-contract",
            document={
                "evaluation_spec_id": "evaluation_spec:test",
                "task_results": [],
                "created_at": "2026-04-21T00:00:00Z",
                "cross_instrument_criteria": {},
            },
        )

    report = EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="evaluation_spec:test",
        evaluation_lane="diagnostic",
        oos_contract_summary={
            "rigor_level": "diagnostic",
            "enforcement": "warn",
            "date_parse": "pass",
            "range_non_overlap": "pass",
            "evaluation_after_execution": "pass",
            "strategy_checkpoint_required": "n/a",
        },
        task_results=(
            EvaluationTaskResult(
                evaluation_lane="diagnostic",
                evaluation_task_id="case:test",
                strategy_id="strategy:test",
            ),
        ),
        created_at="2026-04-21T00:00:00Z",
    )
    restored = EvaluationReport.from_document(
        evaluation_report_id="report:test",
        document=report.to_document(),
    )

    assert restored.evaluation_lane == "diagnostic"
    assert restored.oos_contract_summary["rigor_level"] == "diagnostic"
    assert restored.oos_contract_summary["enforcement"] == "warn"
    assert restored.task_results[0].evaluation_lane == "diagnostic"


def _register_subject_set(store) -> None:
    from alpha_os.portfolio_decision import (
        InstrumentSpec,
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
        UniversePolicySpec,
    )

    store.upsert_subject_set(
        "core_crypto",
        definition=SubjectSet(
            subject_set_id="core_crypto",
            instruments=(
                InstrumentSpec(
                    instrument_id="btc_spot",
                    instrument_type="spot",
                    asset="BTC",
                    asset_class="crypto",
                    cluster="crypto_major",
                ),
            ),
            observation_specs=(
                ObservationSpec(
                    observation_spec_id="btc_close",
                    observable_id="daily_close",
                ),
            ),
            bindings=(
                SubjectObservationBinding(
                    subject_id="BTC_spot",
                    asset="BTC",
                    observation_spec_id="btc_close",
                    instrument_id="btc_spot",
                ),
            ),
            universe_policy=UniversePolicySpec(
                base_currency="USD",
                trading_calendar="24x7",
                benchmark_id="core_crypto",
            ),
        ),
    )


def _register_direct_strategy(store) -> None:
    from alpha_os.trading_strategy import (
        ExecutionPolicySpec,
        RebalanceFrictionPolicySpec,
        StrategyPortfolioSpec,
        TradingStrategyScopeSpec,
        TradingStrategySpec,
        HoldingCostPolicySpec,
    )
    from alpha_os.portfolio_construction_config import (
        PortfolioConstructionSizingSpec,
        PortfolioConstructionSpec,
    )

    store.upsert_trading_strategy(
        trading_strategy=TradingStrategySpec(
            strategy_id="strategy:test",
            label="Test Strategy",
            scope=TradingStrategyScopeSpec(
                subject_set_id="core_crypto",
                target_id="residual_return_3d",
            ),
            signal_discovery_id=None,
            position_rule_id="constant_hold",
            family_mix=None,
            execution_kind="trainless",
            portfolio=StrategyPortfolioSpec(
                portfolio_construction=PortfolioConstructionSpec(
                    sizing_policy=PortfolioConstructionSizingSpec(
                        sizing_method="equal_weight",
                    ),
                    direction_mode="long_only",
                    gross_exposure_cap=1.0,
                ),
                rebalance_friction_policy=RebalanceFrictionPolicySpec(
                    turnover_friction=None,
                    no_trade_band=None,
                ),
                execution_policy=ExecutionPolicySpec(market_impact_bps=None),
                holding_cost_policy=HoldingCostPolicySpec(),
                selection_kind="all_assets",
                top_k=None,
            ),
            created_at="2026-04-20T00:00:00Z",
        )
    )


def _build_direct_evaluation_task():
    from alpha_os.evaluation_task import EvaluationTask

    return EvaluationTask(
        evaluation_task_id="case:test",
        strategy_id="strategy:test",
        evaluation_spec_id="evaluation_spec:test",
    )


def test_store_persists_selected_evaluation_decision_trace(tmp_path):
    from alpha_os.store import EvaluationStore

    store = EvaluationStore(tmp_path / "runtime.db")
    store.ensure_schema()

    store.upsert_evaluation_decision_trace(
        evaluation_report_id="report:test",
        evaluation_task_id="case:test",
        evaluation_fold_label="fold:test",
        evaluation_range_label="range:test",
        result=_build_trace_result(),
        subject_metadata_by_subject={
            "BTC_spot": {
                "asset_class": "crypto",
                "cluster": "crypto_major",
            },
        },
    )

    steps = store.list_evaluation_decision_trace_steps(evaluation_report_id="report:test")
    subject_steps = store.list_evaluation_decision_trace_subject_steps(
        evaluation_report_id="report:test"
    )

    assert len(steps) == 1
    assert steps[0].variant == "selected"
    assert steps[0].step_granularity == "1d"
    assert steps[0].target_id == "residual_return_3d"
    assert steps[0].subject_set_id == "core_crypto"
    assert steps[0].net_return == pytest.approx(0.0087)
    assert steps[0].cost_notional == pytest.approx(0.0013)
    assert len(subject_steps) == 1
    assert subject_steps[0].subject_id == "BTC_spot"
    assert subject_steps[0].asset_class == "crypto"
    assert subject_steps[0].cluster == "crypto_major"
    assert subject_steps[0].entry_allowed is True
    assert subject_steps[0].net_return_contribution == pytest.approx(0.0087)

    store.upsert_evaluation_decision_trace(
        evaluation_report_id="report:test",
        evaluation_task_id="case:test",
        evaluation_fold_label="fold:test",
        evaluation_range_label="range:test",
        result=_build_trace_result().__class__(
            portfolio_id="portfolio:test",
            subject_set_id="core_crypto",
            target_id="residual_return_3d",
            subject_ids=("BTC_spot",),
            steps=(),
        ),
    )

    assert store.list_evaluation_decision_trace_steps(evaluation_report_id="report:test") == []
    assert (
        store.list_evaluation_decision_trace_subject_steps(evaluation_report_id="report:test") == []
    )
    store.close()


def test_evaluation_report_repository_persists_report_with_trace(tmp_path):
    from alpha_os.evaluation_report import EvaluationReport
    from alpha_os.evaluation_report_repository import (
        EvaluationReportRepository,
        PendingEvaluationDecisionTrace,
    )
    from alpha_os.store import EvaluationStore

    store = EvaluationStore(tmp_path / "runtime.db")
    store.ensure_schema()
    repository = EvaluationReportRepository(store)
    report = EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="evaluation_spec:test",
        task_results=(),
        created_at="2026-04-21T00:00:00Z",
    )

    report_state = repository.upsert_report_with_traces(
        report=report,
        pending_decision_traces=(
            PendingEvaluationDecisionTrace(
                evaluation_task_id="case:test",
                evaluation_fold_label="fold:test",
                evaluation_range_label="range:test",
                result=_build_trace_result(),
                subject_metadata_by_subject={
                    "BTC_spot": {
                        "asset_class": "crypto",
                        "cluster": "crypto_major",
                    },
                },
            ),
        ),
    )

    assert report_state.evaluation_report_id == "report:test"
    assert repository.get_report("report:test") is not None
    assert len(repository.list_reports(evaluation_spec_id="evaluation_spec:test")) == 1
    subject_steps = repository.list_decision_trace_subject_steps(evaluation_report_id="report:test")
    assert len(subject_steps) == 1
    assert subject_steps[0].asset_class == "crypto"
    assert subject_steps[0].cluster == "crypto_major"
    store.close()


def test_evaluation_execution_strategy_resolver_selects_execution_strategy():
    from alpha_os.evaluation_execution_strategy import (
        DirectStrategyEvaluationExecutionStrategy,
        PreparedStrategyEvaluationExecutionStrategy,
        evaluation_execution_strategy_for_request,
    )

    assert isinstance(
        evaluation_execution_strategy_for_request(
            SimpleNamespace(input_refs=None)
        ),
        DirectStrategyEvaluationExecutionStrategy,
    )
    assert isinstance(
        evaluation_execution_strategy_for_request(
            SimpleNamespace(input_refs=SimpleNamespace())
        ),
        PreparedStrategyEvaluationExecutionStrategy,
    )


def test_evaluation_runner_persists_direct_selected_trace_without_portfolio_decisions(
    tmp_path,
    monkeypatch,
):
    from alpha_os.evaluation_spec import EvaluationDateRange, EvaluationSpec
    from alpha_os.evaluation_report import EvaluationMetricGroupResult
    from alpha_os.evaluation_runner import EvaluationRunRequest, evaluate_evaluation_spec_state
    from alpha_os.signal_discovery_strategy_evaluation import (
        EvaluationTraceRangeResult,
        StrategyEvaluationResult,
    )
    from alpha_os.store import EvaluationStore

    store = EvaluationStore(tmp_path / "runtime.db")
    store.ensure_schema()
    _register_subject_set(store)
    _register_direct_strategy(store)
    evaluation_spec_state = store.upsert_evaluation_spec(
        "evaluation_spec:test",
        definition=EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="fold:test",
                start_date="2026-01-01",
                end_date="2026-01-02",
            ),
            evaluation_date_ranges=(
                EvaluationDateRange(
                    label="range:test",
                    start_date="2026-01-01",
                    end_date="2026-01-02",
                ),
            ),
            metric_group_names=("decision_quality",),
            target_ids=("residual_return_3d",),
            metric_windows=(2,),
        ),
        recorded_at="2026-04-20T00:00:00Z",
    )

    def _fake_direct_case(**kwargs):
        return StrategyEvaluationResult(
            metric_group_results_by_name={
                "decision_quality": EvaluationMetricGroupResult(
                    metric_group_name="decision_quality",
                    source="native_plan",
                    metrics={
                        "mean_decision_net_return": 0.0087,
                        "mean_decision_step_count": 1.0,
                    },
                )
            },
            failure_finding_groups=(),
            selected_trace_results=(
                EvaluationTraceRangeResult(
                    range_label="range:test",
                    result=_build_trace_result(),
                ),
            ),
        )

    monkeypatch.setattr(
        "alpha_os.evaluation_execution_strategy.run_strategy_backtest_from_store",
        _fake_direct_case,
    )

    report_state = evaluate_evaluation_spec_state(
        EvaluationRunRequest(
            store=store,
            default_target_id="residual_return_3d",
            evaluation_spec_state=evaluation_spec_state,
            evaluation_tasks=(_build_direct_evaluation_task(),),
            base_url="http://example.com",
        )
    )

    steps = store.list_evaluation_decision_trace_steps(
        evaluation_report_id=report_state.evaluation_report_id
    )
    subject_steps = store.list_evaluation_decision_trace_subject_steps(
        evaluation_report_id=report_state.evaluation_report_id
    )

    assert len(steps) == 1
    assert steps[0].evaluation_task_id == "case:test"
    assert steps[0].evaluation_fold_label == "fold:test"
    assert steps[0].evaluation_range_label == "range:test"
    assert len(subject_steps) == 1
    assert subject_steps[0].asset_class == "crypto"
    assert subject_steps[0].cluster == "crypto_major"
    assert store.list_portfolio_decisions(limit=10) == []
    store.close()


def test_decision_trace_diagnostics_uses_exact_trace_rows(tmp_path):
    from alpha_os.evaluation_decision_trace_diagnostics import (
        build_evaluation_decision_trace_diagnostics,
    )
    from alpha_os.store import EvaluationStore

    store = EvaluationStore(tmp_path / "runtime.db")
    store.ensure_schema()
    _persist_report(store)
    store.upsert_evaluation_decision_trace(
        evaluation_report_id="report:test",
        evaluation_task_id="case:test",
        evaluation_fold_label="fold:test",
        evaluation_range_label="2017",
        result=_build_tail_risk_trace_result(),
        subject_metadata_by_subject={
            "EQ_future": {
                "asset_class": "equity_index",
                "cluster": "eq_us",
            },
            "TY_future": {
                "asset_class": "rates",
                "cluster": "rates_us",
            },
        },
    )

    report = build_evaluation_decision_trace_diagnostics(
        store,
        evaluation_report_id="report:test",
        range_labels=("2017",),
        top_n=2,
    )

    assert report.evaluation_lane == "diagnostic"
    range_result = report.ranges[0]
    assert range_result.range_label == "2017"
    baseline = range_result.baseline
    assert baseline.step_count == 2
    assert baseline.subject_step_count == 4
    assert baseline.direction.hit_rate == pytest.approx(0.5)
    assert baseline.direction.signed_edge == pytest.approx(-0.00575)
    long_row = baseline.direction.rows[0]
    short_row = baseline.direction.rows[1]
    assert long_row.direction == "long"
    assert long_row.hit_rate == pytest.approx(0.5)
    assert long_row.wrong_way_pnl_notional == pytest.approx(-0.06)
    assert short_row.direction == "short"
    assert short_row.hit_rate == pytest.approx(0.5)
    assert short_row.wrong_way_pnl_notional == pytest.approx(-0.008)
    cost = baseline.cost_turnover
    assert cost.gross_return == pytest.approx(-0.01888)
    assert cost.net_return == pytest.approx(-0.02185075)
    assert cost.cost_notional == pytest.approx(0.003)
    assert cost.execution_cost_notional == pytest.approx(0.003)
    assert cost.total_turnover == pytest.approx(1.0)
    assert cost.average_turnover == pytest.approx(0.5)
    assert cost.cost_per_traded_notional == pytest.approx(0.003)
    assert cost.cost_to_abs_gross_pnl == pytest.approx(0.1666666667)
    assert baseline.exposure.max_subject_concentration_label == "EQ_future"
    assert baseline.exposure.max_subject_concentration == pytest.approx(0.6)
    assert baseline.contribution.subject_rows[0].label == "EQ_future"
    assert baseline.contribution.cluster_rows[0].label == "eq_us"
    assert baseline.contribution.asset_class_rows[0].label == "equity_index"
    tail_risk = range_result.tail_risk
    assert tail_risk.step_count == 2
    assert tail_risk.subject_step_count == 4
    assert tail_risk.worst_day == "2017-01-04"
    assert tail_risk.cost_notional == pytest.approx(0.003)
    assert tail_risk.subject_losers[0].label == "EQ_future"
    assert tail_risk.subject_losers[0].net_pnl_notional == pytest.approx(-0.032)
    assert tail_risk.cluster_losers[0].label == "eq_us"
    assert tail_risk.direction_rows[0].direction == "long"
    assert tail_risk.direction_rows[0].wrong_way_pnl_notional == pytest.approx(-0.06)
    assert tail_risk.exposure.max_subject_concentration_label == "EQ_future"
    assert tail_risk.exposure.max_subject_concentration == pytest.approx(0.6)
    modes = {item.mode: item for item in range_result.direction_ablation.modes}
    assert set(modes) == {"long_short", "long_only", "short_only"}
    assert modes["long_short"].net_return == pytest.approx(-0.02185075)
    assert modes["long_only"].net_return == pytest.approx(-0.033769)
    assert modes["short_only"].net_return == pytest.approx(0.01083425)
    assert modes["long_only"].average_short_exposure == pytest.approx(0.0)
    assert modes["short_only"].average_long_exposure == pytest.approx(0.0)
    assert modes["long_only"].asset_class_rows[0].label == "equity_index"
    assert modes["short_only"].asset_class_rows[0].label == "rates"
    store.close()


def test_decision_trace_diagnostics_defaults_to_trace_ranges(tmp_path):
    from alpha_os.evaluation_decision_trace_diagnostics import (
        build_evaluation_decision_trace_diagnostics,
    )
    from alpha_os.store import EvaluationStore

    store = EvaluationStore(tmp_path / "runtime.db")
    store.ensure_schema()
    _persist_report(store)
    store.upsert_evaluation_decision_trace(
        evaluation_report_id="report:test",
        evaluation_task_id="case:test",
        evaluation_fold_label="fold:test",
        evaluation_range_label="2017",
        result=_build_tail_risk_trace_result(),
    )

    report = build_evaluation_decision_trace_diagnostics(
        store,
        evaluation_report_id="report:test",
    )

    assert tuple(item.range_label for item in report.ranges) == ("2017",)
    store.close()


def test_decision_trace_diagnostics_requires_persisted_trace(tmp_path):
    from alpha_os.evaluation_decision_trace_diagnostics import (
        build_evaluation_decision_trace_diagnostics,
    )
    from alpha_os.store import EvaluationStore

    store = EvaluationStore(tmp_path / "runtime.db")
    store.ensure_schema()
    _persist_report(store)

    with pytest.raises(ValueError, match="trace is missing"):
        build_evaluation_decision_trace_diagnostics(
            store,
            evaluation_report_id="report:test",
            range_labels=("2017",),
        )
    store.close()


def test_decision_trace_diagnostics_cli_prints_and_writes_text(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    out_path = tmp_path / "diagnostics.txt"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    _persist_report(store)
    store.upsert_evaluation_decision_trace(
        evaluation_report_id="report:test",
        evaluation_task_id="case:test",
        evaluation_fold_label="fold:test",
        evaluation_range_label="2017",
        result=_build_tail_risk_trace_result(),
        subject_metadata_by_subject={
            "EQ_future": {
                "asset_class": "equity_index",
                "cluster": "eq_us",
            },
            "TY_future": {
                "asset_class": "rates",
                "cluster": "rates_us",
            },
        },
    )
    store.close()

    assert (
        main(
            [
                "show-evaluation-diagnostics",
                "--db",
                str(db_path),
                "--range-label",
                "2017",
                "--top-n",
                "2",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "alpha-os evaluation diagnostics" in output
    assert "Lane:     diagnostic" in output
    assert "baseline:" in output
    assert "tail_risk:" in output
    assert "direction_ablation:" in output
    assert "hit_rate=0.500000" in output
    assert "cost_turnover" in output
    assert "EQ_future" in output
    assert "wrong_way=-0.060000" in output
    assert "mode=long_only" in output
    assert "mode=short_only" in output

    assert (
        main(
            [
                "show-evaluation-diagnostics",
                "--db",
                str(db_path),
                "--range-label",
                "2017",
                "--output",
                str(out_path),
            ]
        )
        == 0
    )
    assert "Wrote evaluation diagnostics" in capsys.readouterr().out
    written = out_path.read_text()
    assert "trace contribution ablation" in written
    assert "direction_contribution" in written
    assert "cluster_losers" in out_path.read_text()


def test_diagnostics_cli_replaces_legacy_diagnostic_commands():
    from alpha_os.cli import build_cli_parser

    help_text = build_cli_parser().format_help()

    assert "show-diagnostics" in help_text
    assert "show-evaluation-diagnostics" not in help_text
    assert "show-evaluation-baseline-diagnostics" not in help_text
    assert "show-evaluation-tail-risk-attribution" not in help_text
    assert "show-evaluation-direction-ablation" not in help_text
