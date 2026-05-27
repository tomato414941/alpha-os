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
    from alpha_os.evaluation_cost_config import TradingEnvironment
    from alpha_os.trading_strategy import (
        StrategyPortfolioSpec,
        TradingStrategySpec,
    )
    from alpha_os.portfolio_construction_config import (
        PortfolioConstructionSizingSpec,
        PortfolioConstructionSpec,
    )

    store.upsert_trading_strategy(
        trading_strategy=TradingStrategySpec(
            strategy_id="strategy:test",
            label="Test Strategy",
            subject_set_id="core_crypto",
            target_id="residual_return_3d",
            signal_discovery_id=None,
            position_rule_id="constant_hold",
            family_mix=None,
            portfolio=StrategyPortfolioSpec(
                portfolio_construction=PortfolioConstructionSpec(
                    sizing_policy=PortfolioConstructionSizingSpec(
                        sizing_method="equal_weight",
                    ),
                    direction_mode="long_only",
                    gross_exposure_cap=1.0,
                ),
                trading_environment=TradingEnvironment(),
                selection_kind="all_assets",
                top_k=None,
            ),
            created_at="2026-04-20T00:00:00Z",
        )
    )


def _build_direct_evaluation_case():
    return ("case:test", "strategy:test")


def test_evaluation_runner_persists_direct_report_without_portfolio_decisions(
    tmp_path,
    monkeypatch,
):
    from alpha_os.evaluation_spec import EvaluationDateRange, EvaluationSpec
    from alpha_os.evaluation_result import EvaluationMetricGroupResult
    from alpha_os.evaluation_runner import evaluate_evaluation_spec_state
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

    run_result_state = evaluate_evaluation_spec_state(
        store=store,
        evaluation_spec_state=evaluation_spec_state,
        evaluation_targets=(_build_direct_evaluation_case(),),
        base_url="http://example.com",
    )

    assert run_result_state.evaluation_run_result_id.startswith("evaluation_spec:test:")
    assert len(run_result_state.run_result.results) == 1
    assert tuple(run_result_state.run_result.results) == ("case:test",)
    assert store.list_portfolio_decisions(limit=10) == []
    store.close()
