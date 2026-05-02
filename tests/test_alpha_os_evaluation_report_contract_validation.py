from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest


def _register_default_validation_signal_specs(store) -> None:
    for signal_id in ("momentum_1d", "reversal_1d"):
        store.register_signal_spec(signal_id=signal_id)


def _register_singleton_subject_set(store, *, subject_set_id: str = "core_crypto") -> None:
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
        UniversePolicySpec,
    )

    store.upsert_subject_set(
        subject_set_id,
        definition=SubjectSet(
            subject_set_id=subject_set_id,
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
                ),
            ),
            universe_policy=UniversePolicySpec(
                base_currency="USD",
                trading_calendar="24x7",
                benchmark_id=subject_set_id,
            ),
        ),
    )


def _register_multi_subject_set(
    store,
    *,
    subject_set_id: str = "macro_pair",
    complete_universe_policy: bool = False,
) -> None:
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
        UniversePolicySpec,
    )

    store.upsert_subject_set(
        subject_set_id,
        definition=SubjectSet(
            subject_set_id=subject_set_id,
            observation_specs=(
                ObservationSpec(
                    observation_spec_id="btc_close",
                    observable_id="daily_close",
                ),
                ObservationSpec(
                    observation_spec_id="eth_close",
                    observable_id="daily_close",
                ),
            ),
            bindings=(
                SubjectObservationBinding(
                    subject_id="BTC_spot",
                    asset="BTC",
                    observation_spec_id="btc_close",
                ),
                SubjectObservationBinding(
                    subject_id="ETH_spot",
                    asset="ETH",
                    observation_spec_id="eth_close",
                ),
            ),
            universe_policy=(
                UniversePolicySpec(
                    base_currency="USD",
                    trading_calendar="24x7",
                    benchmark_id=subject_set_id,
                )
                if complete_universe_policy
                else UniversePolicySpec(
                    base_currency="USD",
                    trading_calendar=None,
                    benchmark_id=None,
                )
            ),
        ),
    )


def _register_direct_strategy(
    store,
    *,
    strategy_id: str = "strategy:test",
    subject_set_id: str = "core_crypto",
) -> None:
    from alpha_os.trading_strategy import (
        ExecutionPolicySpec,
        HoldingCostPolicySpec,
        PortfolioPolicySpec,
        RebalanceFrictionPolicySpec,
        RebalancePolicySpec,
        RiskPolicySpec,
        SelectionPolicySpec,
        StrategyPortfolioSpec,
        SizingPolicySpec,
        TradingStrategyScopeSpec,
        TradingStrategySpec,
    )

    portfolio_policy = PortfolioPolicySpec(
        selection_policy=SelectionPolicySpec(
            selection_kind="top_k",
            top_k=3,
        ),
        sizing_policy=SizingPolicySpec(sizing_method="equal_weight"),
        rebalance_policy=RebalancePolicySpec(rebalance="every_1_steps"),
        risk_policy=RiskPolicySpec(
            long_only=True,
            gross_exposure_cap=1.0,
            target_vol=0.12,
            gross_leverage_cap=1.5,
            net_exposure_target=0.3,
        ),
    )
    store.upsert_trading_strategy(
        trading_strategy=TradingStrategySpec(
            strategy_id=strategy_id,
            label="Test Strategy",
            scope=TradingStrategyScopeSpec(
                subject_set_id=subject_set_id,
                target_id="residual_return_3d",
            ),
            signal_discovery_id=None,
            position_rule_id="constant_hold",
            family_mix=None,
            execution_kind="trainless",
            portfolio=StrategyPortfolioSpec.from_legacy(
                portfolio_policy=portfolio_policy,
                rebalance_friction_policy=RebalanceFrictionPolicySpec(
                    turnover_friction=0.1,
                    no_trade_band=0.02,
                    execution_cost_aversion=1.0,
                ),
                execution_policy=ExecutionPolicySpec(
                    market_impact_bps=5.0,
                    fee_bps=2.0,
                    bid_ask_spread_bps=3.0,
                ),
                holding_cost_policy=HoldingCostPolicySpec(
                    funding_bps_per_step=1.5,
                    borrow_fee_bps_per_step=2.5,
                ),
                portfolio_construction=None,
                sleeve_composition=None,
            ),
            created_at="2026-04-18T00:00:00Z",
        )
    )


def _build_direct_evaluation_task():
    from alpha_os.evaluation_task import EvaluationTask

    return EvaluationTask(
        evaluation_task_id="case:test",
        strategy_id="strategy:test",
        evaluation_spec_id="evaluation_spec:test",
    )


def _build_direct_evaluation_task_for_strategy(strategy_id: str):
    return replace(_build_direct_evaluation_task(), strategy_id=strategy_id)


def test_evaluation_report_contract_validation_passes_for_structured_validation_and_report(tmp_path):
    import alpha_os.validation_service as validation_service
    from alpha_os.evaluation_report import (
        EvaluationTaskResult,
        EvaluationMetricGroupResult,
        EvaluationReport,
        EvaluationFailureFinding,
        EvaluationFailureFindingGroup,
    )
    from alpha_os.evaluation_report_contract_validation import validate_evaluation_report_contract
    from alpha_os.store import EvaluationStore
    from alpha_os.validation_service import run_validation
    from alpha_os.validation_spec import ValidationDateRange, ValidationSpec

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    _register_default_validation_signal_specs(store)
    _register_singleton_subject_set(store)

    frame = {
        "timestamp": [
            "2026-03-20T00:00:00Z",
            "2026-03-21T00:00:00Z",
            "2026-03-22T00:00:00Z",
            "2026-03-23T00:00:00Z",
            "2026-03-24T00:00:00Z",
            "2026-03-25T00:00:00Z",
        ],
        "value": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
    }

    def _fake_loader(*, base_url: str, asset: str, observation_spec):
        import pandas as pd

        return pd.DataFrame(frame)

    original_loader = validation_service._load_price_frame_from_signal_noise
    validation_service._load_price_frame_from_signal_noise = _fake_loader
    try:
        result = run_validation(
            store,
            spec=ValidationSpec(
                signal_ids=("momentum_1d", "reversal_1d"),
                target_ids=("residual_return_1d", "residual_return_3d"),
                date_ranges=(
                    ValidationDateRange(
                        label="mini",
                        start_date="2026-03-22",
                        end_date="2026-03-22",
                    ),
                ),
                metric_windows=(20,),
                aggregation_kinds=("active_equal_mean", "corr_weighted_mean"),
                subject_set_ids=("core_crypto",),
                base_url="http://example.com",
            ),
            recorded_at="2026-03-29T00:00:00+00:00",
        )
    finally:
        validation_service._load_price_frame_from_signal_noise = original_loader
        store.close()

    store = EvaluationStore(db_path)
    try:
        validation_run = store.get_validation_run(result.run_id)
        assert validation_run is not None
    finally:
        store.close()

    report = EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="evaluation_spec:test",
        task_results=(
            EvaluationTaskResult(
                evaluation_task_id="case:test",
                strategy_id="strategy:test",
                signal_discovery_id="signal-discovery:test",
                strategy_contract_fields={
                    "selection": "top_k",
                    "top_k": 3,
                    "sizing": "equal_weight",
                    "rebalance": "every_1_steps",
                    "long_only": "true",
                    "gross_exposure_cap": 1.0,
                    "target_vol": 0.12,
                    "gross_leverage_cap": 1.5,
                    "net_exposure_target": 0.3,
                    "turnover_friction": 0.1,
                    "no_trade_band": 0.02,
                    "market_impact_bps": 5.0,
                    "fee_bps": 2.0,
                    "constraint_stages": (
                        "sizing_time:target_vol;"
                        "post_sizing_normalization:long_only,gross_exposure_cap,"
                        "gross_leverage_cap,net_exposure_target"
                    ),
                    "funding_bps_per_step": 1.5,
                    "borrow_fee_bps_per_step": 2.5,
                    "subject_set": "core_crypto",
                    "base_currency": "USD",
                    "trading_calendar": "24x7",
                    "benchmark_id": "core_crypto",
                },
                subject_set_facts="bindings=1 instruments=0 subject_kinds=asset instrument_types=- contract_groups=instrument,observation_spec,binding,universe_policy",
                subject_set_contract_groups=(
                    "instrument",
                    "observation_spec",
                    "binding",
                    "universe_policy",
                ),
                universe_policy_fields={
                    "base_currency": "USD",
                    "trading_calendar": "24x7",
                    "benchmark_id": "core_crypto",
                },
                constraint_stages=(
                    "sizing_time:target_vol",
                    "post_sizing_normalization:long_only,gross_exposure_cap,gross_leverage_cap,net_exposure_target",
                ),
                metric_group_results=(
                    EvaluationMetricGroupResult(
                        metric_group_name="signed_belief_quality",
                        source="native_plan",
                        metrics={
                            "mean_survivor_corr": 0.10,
                            "mean_survivor_stability_score": 0.20,
                            "mean_component_confidence": 0.30,
                            "mean_range_signed_belief_corr": 0.15,
                            "best_range_signed_belief_corr": 0.20,
                            "worst_range_signed_belief_corr": 0.05,
                        },
                    ),
                    EvaluationMetricGroupResult(
                        metric_group_name="portfolio_target_return_alignment",
                        source="native_plan",
                        metrics={
                            "mean_range_portfolio_target_return_corr": 0.12,
                            "best_range_portfolio_target_return_corr": 0.20,
                            "worst_range_portfolio_target_return_corr": 0.01,
                        },
                    ),
                    EvaluationMetricGroupResult(
                        metric_group_name="decision_quality",
                        source="native_plan",
                        metrics={
                            "mean_decision_net_return": 0.12,
                            "best_decision_net_return": 0.20,
                            "mean_decision_drawdown": 0.05,
                            "mean_decision_turnover": 0.10,
                            "mean_decision_gross_leverage_exposure": 0.80,
                            "mean_decision_net_leverage_exposure": 0.10,
                            "mean_decision_long_leverage_exposure": 0.45,
                            "mean_decision_short_leverage_exposure": 0.35,
                            "mean_decision_gross_notional_exposure": 0.80,
                            "mean_decision_net_notional_exposure": 0.10,
                            "mean_decision_long_notional_exposure": 0.45,
                            "mean_decision_short_notional_exposure": 0.35,
                            "mean_decision_traded_notional": 0.20,
                            "total_decision_cost_notional": 0.01,
                            "total_decision_funding_cost_notional": 0.004,
                            "total_decision_borrow_cost_notional": 0.003,
                            "total_decision_roll_cost_notional": 0.002,
                            "mean_decision_step_count": 12.0,
                            "total_decision_step_count": 12,
                            "mean_step_net_return": 0.01,
                            "step_net_return_std": 0.02,
                            "pooled_step_max_drawdown": 0.06,
                            "annualized_step_sharpe": 1.20,
                        },
                    ),
                    EvaluationMetricGroupResult(
                        metric_group_name="sizing_policy_quality",
                        source="native_plan",
                        metrics={
                            "selected_sizing_method": "signal_weighted",
                            "selected_sizing_engine": "rule_based",
                            "mean_equal_weight_decision_net_return": 0.09,
                            "mean_equal_weight_daily_decision_net_return": 0.08,
                            "mean_selected_vs_equal_weight_decision_net_return_edge": 0.03,
                            "best_selected_vs_equal_weight_decision_net_return_edge": 0.05,
                            "worst_selected_vs_equal_weight_decision_net_return_edge": -0.01,
                            "mean_daily_signal_weighted_vs_equal_weight_decision_net_return_edge": 0.02,
                            "mean_selected_vs_equal_weight_drawdown_edge": 0.01,
                            "mean_selected_vs_equal_weight_turnover_edge": 0.04,
                        },
                    ),
                    EvaluationMetricGroupResult(
                        metric_group_name="rebalance_policy_quality",
                        source="native_plan",
                        metrics={
                            "selected_rebalance_interval_steps": 1,
                            "mean_selected_vs_daily_rebalance_net_return_edge": 0.01,
                            "best_selected_vs_daily_rebalance_net_return_edge": 0.03,
                            "worst_selected_vs_daily_rebalance_net_return_edge": -0.02,
                            "mean_selected_vs_daily_rebalance_turnover_savings": 0.05,
                            "mean_equal_weight_vs_daily_rebalance_net_return_edge": 0.02,
                            "mean_equal_weight_vs_daily_rebalance_turnover_savings": 0.04,
                        },
                    ),
                    EvaluationMetricGroupResult(
                        metric_group_name="robustness",
                        source="native_plan",
                        metrics={
                            "range_count": 2,
                            "signed_belief_corr_std": 0.01,
                            "portfolio_target_return_corr_std": 0.02,
                            "decision_net_return_std": 0.03,
                            "decision_negative_fraction": 0.0,
                            "worst_decision_net_return": 0.04,
                        },
                    ),
                ),
                failure_finding_groups=(
                    EvaluationFailureFindingGroup(
                        metric_group_name="decision_quality",
                        source="native_plan",
                        findings=(EvaluationFailureFinding(label="tail", severity=0.25, metrics={}),),
                    ),
                    EvaluationFailureFindingGroup(
                        metric_group_name="sizing_policy_quality",
                        source="native_plan",
                        findings=(),
                    ),
                    EvaluationFailureFindingGroup(
                        metric_group_name="rebalance_policy_quality",
                        source="native_plan",
                        findings=(),
                    ),
                    EvaluationFailureFindingGroup(
                        metric_group_name="signed_belief_quality",
                        source="native_plan",
                        findings=(),
                    ),
                    EvaluationFailureFindingGroup(
                        metric_group_name="portfolio_target_return_alignment",
                        source="native_plan",
                        findings=(),
                    ),
                ),
            ),
        ),
        created_at="2026-04-18T00:00:00Z",
    )

    result = validate_evaluation_report_contract(
        validation_run=validation_run,
        evaluation_report=report,
    )

    assert result.passed is True
    assert result.issues == ()


def test_evaluation_report_contract_validation_passes_for_persisted_current_paths(tmp_path, monkeypatch):
    from alpha_os.evaluation_runner import EvaluationRunRequest, evaluate_evaluation_spec_state
    from alpha_os.evaluation_spec import (
        EvaluationDateRange as EvaluationSpecDateRange,
        EvaluationSpec,
    )
    from alpha_os.evaluation_report_contract_validation import validate_evaluation_report_contract
    from alpha_os.evaluation_report import (
        EvaluationMetricGroupResult,
        EvaluationFailureFinding,
        EvaluationFailureFindingGroup,
    )
    from alpha_os.store import EvaluationStore
    from alpha_os.validation_service import run_validation
    from alpha_os.validation_spec import (
        ValidationDateRange as ValidationWindow,
        ValidationSpec,
    )

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    _register_default_validation_signal_specs(store)
    _register_singleton_subject_set(store)
    _register_direct_strategy(store)

    frame = {
        "timestamp": [
            "2026-03-20T00:00:00Z",
            "2026-03-21T00:00:00Z",
            "2026-03-22T00:00:00Z",
            "2026-03-23T00:00:00Z",
            "2026-03-24T00:00:00Z",
            "2026-03-25T00:00:00Z",
        ],
        "value": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
    }

    def _fake_loader(*, base_url: str, asset: str, observation_spec):
        import pandas as pd

        return pd.DataFrame(frame)

    def _fake_direct_case(**kwargs):
        return (
            {
                "decision_quality": EvaluationMetricGroupResult(
                    metric_group_name="decision_quality",
                    source="native_plan",
                    metrics={
                        "mean_decision_net_return": 0.12,
                        "best_decision_net_return": 0.20,
                        "mean_decision_drawdown": 0.05,
                        "mean_decision_turnover": 0.10,
                        "mean_decision_gross_leverage_exposure": 0.80,
                        "mean_decision_net_leverage_exposure": 0.10,
                        "mean_decision_long_leverage_exposure": 0.45,
                        "mean_decision_short_leverage_exposure": 0.35,
                        "mean_decision_gross_notional_exposure": 0.80,
                        "mean_decision_net_notional_exposure": 0.10,
                        "mean_decision_long_notional_exposure": 0.45,
                        "mean_decision_short_notional_exposure": 0.35,
                        "mean_decision_traded_notional": 0.20,
                        "total_decision_cost_notional": 0.01,
                        "total_decision_funding_cost_notional": 0.004,
                        "total_decision_borrow_cost_notional": 0.003,
                        "total_decision_roll_cost_notional": 0.002,
                        "mean_decision_step_count": 12.0,
                        "total_decision_step_count": 12,
                        "mean_step_net_return": 0.01,
                        "step_net_return_std": 0.02,
                        "pooled_step_max_drawdown": 0.06,
                        "annualized_step_sharpe": 1.20,
                    },
                ),
            },
            (
                EvaluationFailureFindingGroup(
                    metric_group_name="decision_quality",
                    source="native_plan",
                    findings=(EvaluationFailureFinding(label="tail", severity=0.25, metrics={}),),
                ),
            ),
        )

    evaluation_spec_state = store.upsert_evaluation_spec(
        "evaluation_spec:test",
        definition=EvaluationSpec(
            execution_range=EvaluationSpecDateRange(
                label="eval",
                start_date="2026-03-20",
                end_date="2026-03-25",
            ),
            metric_group_names=("decision_quality",),
            target_ids=("residual_return_3d",),
            metric_windows=(20,),
        ),
        recorded_at="2026-04-18T00:00:00Z",
    )

    monkeypatch.setattr(
        "alpha_os.validation_service._load_price_frame_from_signal_noise",
        _fake_loader,
    )
    monkeypatch.setattr(
        "alpha_os.evaluation_execution_strategy.evaluate_trainless_candidate_backtest",
        _fake_direct_case,
    )

    validation_result = run_validation(
        store,
        spec=ValidationSpec(
            signal_ids=("momentum_1d", "reversal_1d"),
            target_ids=("residual_return_1d", "residual_return_3d"),
            date_ranges=(
                ValidationWindow(
                    label="mini",
                    start_date="2026-03-22",
                    end_date="2026-03-22",
                ),
            ),
            metric_windows=(20,),
            aggregation_kinds=("active_equal_mean", "corr_weighted_mean"),
            subject_set_ids=("core_crypto",),
            base_url="http://example.com",
        ),
        recorded_at="2026-03-29T00:00:00+00:00",
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
    store.close()

    store = EvaluationStore(db_path)
    try:
        validation_run = store.get_validation_run(validation_result.run_id)
        persisted_report = store.get_evaluation_report(report_state.evaluation_report_id)
        assert validation_run is not None
        assert persisted_report is not None
    finally:
        store.close()

    result = validate_evaluation_report_contract(
        validation_run=validation_run,
        evaluation_report=persisted_report,
    )

    assert result.passed is True
    assert result.issues == ()


def test_evaluation_report_contract_validation_detects_universe_policy_mismatch(tmp_path):
    import alpha_os.validation_service as validation_service
    from alpha_os.evaluation_report import EvaluationTaskResult, EvaluationReport
    from alpha_os.evaluation_report_contract_validation import validate_evaluation_report_contract
    from alpha_os.store import EvaluationStore
    from alpha_os.validation_service import run_validation
    from alpha_os.validation_spec import ValidationDateRange, ValidationSpec

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    _register_default_validation_signal_specs(store)
    _register_singleton_subject_set(store)

    frame = {
        "timestamp": [
            "2026-03-20T00:00:00Z",
            "2026-03-21T00:00:00Z",
            "2026-03-22T00:00:00Z",
            "2026-03-23T00:00:00Z",
            "2026-03-24T00:00:00Z",
            "2026-03-25T00:00:00Z",
        ],
        "value": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
    }

    def _fake_loader(*, base_url: str, asset: str, observation_spec):
        import pandas as pd

        return pd.DataFrame(frame)

    original_loader = validation_service._load_price_frame_from_signal_noise
    validation_service._load_price_frame_from_signal_noise = _fake_loader
    try:
        result = run_validation(
            store,
            spec=ValidationSpec(
                signal_ids=("momentum_1d", "reversal_1d"),
                target_ids=("residual_return_1d", "residual_return_3d"),
                date_ranges=(
                    ValidationDateRange(
                        label="mini",
                        start_date="2026-03-22",
                        end_date="2026-03-22",
                    ),
                ),
                metric_windows=(20,),
                aggregation_kinds=("active_equal_mean",),
                subject_set_ids=("core_crypto",),
                base_url="http://example.com",
            ),
            recorded_at="2026-03-29T00:00:00+00:00",
        )
    finally:
        validation_service._load_price_frame_from_signal_noise = original_loader
        store.close()

    store = EvaluationStore(db_path)
    try:
        validation_run = store.get_validation_run(result.run_id)
        assert validation_run is not None
    finally:
        store.close()

    report = EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="evaluation_spec:test",
        task_results=(
            EvaluationTaskResult(
                evaluation_task_id="case:test",
                strategy_id="strategy:test",
                strategy_contract_fields={
                    "subject_set": "core_crypto",
                    "base_currency": "EUR",
                    "trading_calendar": "24x7",
                    "benchmark_id": "core_crypto",
                },
                subject_set_facts="bindings=1 instruments=0 contract_groups=instrument,observation_spec,binding,universe_policy",
                subject_set_contract_groups=(
                    "instrument",
                    "observation_spec",
                    "binding",
                    "universe_policy",
                ),
                universe_policy_fields={
                    "base_currency": "EUR",
                    "trading_calendar": "24x7",
                    "benchmark_id": "core_crypto",
                },
                constraint_stages=("sizing_time:target_vol",),
            ),
        ),
        created_at="2026-04-18T00:00:00Z",
    )

    result = validate_evaluation_report_contract(
        validation_run=validation_run,
        evaluation_report=report,
    )

    assert result.passed is False
    assert (
        "evaluation report task result case:test universe-policy fields do not match validation result set for core_crypto"
        in result.issues
    )
    assert (
        "evaluation report task result case:test strategy contract is missing universe-policy field base_currency"
        not in result.issues
    )


def test_evaluate_evaluation_spec_state_rejects_incomplete_universe_policy_for_direct_case(
    tmp_path, monkeypatch
):
    from alpha_os.evaluation_runner import EvaluationRunRequest, evaluate_evaluation_spec_state
    from alpha_os.evaluation_spec import (
        EvaluationDateRange as EvaluationSpecDateRange,
        EvaluationSpec,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    _register_multi_subject_set(store, complete_universe_policy=False)
    _register_direct_strategy(
        store,
        strategy_id="strategy:multi",
        subject_set_id="macro_pair",
    )
    evaluation_spec_state = store.upsert_evaluation_spec(
        "evaluation_spec:test",
        definition=EvaluationSpec(
            execution_range=EvaluationSpecDateRange(
                label="eval",
                start_date="2026-03-20",
                end_date="2026-03-25",
            ),
            metric_group_names=("decision_quality",),
            target_ids=("residual_return_3d",),
            metric_windows=(20,),
        ),
        recorded_at="2026-04-18T00:00:00Z",
    )

    def _should_not_run(**kwargs):
        raise AssertionError("direct evaluation should not run for invalid universe")

    monkeypatch.setattr(
        "alpha_os.evaluation_execution_strategy.evaluate_trainless_candidate_backtest",
        _should_not_run,
    )

    try:
        with pytest.raises(ValueError, match="subject set universe policy is incomplete"):
            evaluate_evaluation_spec_state(
                    EvaluationRunRequest(
                        store=store,
                        default_target_id="residual_return_3d",
                        evaluation_spec_state=evaluation_spec_state,
                        evaluation_tasks=(
                            _build_direct_evaluation_task_for_strategy("strategy:multi"),
                        ),
                        base_url="http://example.com",
                    )
                )
    finally:
        store.close()


def test_frozen_survivor_snapshots_reject_incomplete_universe_policy(tmp_path):
    from alpha_os.evaluation_runner import generate_frozen_survivor_test_snapshots
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    _register_multi_subject_set(store, complete_universe_policy=False)

    try:
        with pytest.raises(ValueError, match="subject set universe policy is incomplete"):
            generate_frozen_survivor_test_snapshots(
                store,
                subject_set_id="macro_pair",
                survivor_signal_ids=[],
                start_date="2026-03-20",
                end_date="2026-03-25",
                base_url="http://example.com",
            )
    finally:
        store.close()


def test_frozen_snapshot_start_date_uses_fixture_daily_calendar_days():
    from alpha_os.evaluation_runner import frozen_snapshot_start_date
    from alpha_os.evaluation_spec import EvaluationDateRange
    from alpha_os.portfolio_construction_config import PortfolioConstructionSpec
    from alpha_os.signal_registry import SignalDefinition
    from alpha_os.targets import residual_return_target_definition

    start_date = frozen_snapshot_start_date(
        evaluation_date_ranges=(
            EvaluationDateRange(
                label="test",
                start_date="2026-01-12",
                end_date="2026-01-16",
            ),
        ),
        executable_definitions=[
            SignalDefinition(
                signal_id="signal:test",
                kind="trend",
                lookback=3,
                target=residual_return_target_definition(3),
            )
        ],
        metric_window=3,
        portfolio_construction=PortfolioConstructionSpec(),
        trading_calendar="fixture_daily",
    )

    assert start_date == "2026-01-09"


def test_frozen_snapshot_start_date_uses_business_days_by_default():
    from alpha_os.evaluation_runner import frozen_snapshot_start_date
    from alpha_os.evaluation_spec import EvaluationDateRange
    from alpha_os.portfolio_construction_config import PortfolioConstructionSpec
    from alpha_os.signal_registry import SignalDefinition
    from alpha_os.targets import residual_return_target_definition

    start_date = frozen_snapshot_start_date(
        evaluation_date_ranges=(
            EvaluationDateRange(
                label="test",
                start_date="2026-01-12",
                end_date="2026-01-16",
            ),
        ),
        executable_definitions=[
            SignalDefinition(
                signal_id="signal:test",
                kind="trend",
                lookback=3,
                target=residual_return_target_definition(3),
            )
        ],
        metric_window=3,
        portfolio_construction=PortfolioConstructionSpec(),
        trading_calendar="business_day",
    )

    assert start_date == "2026-01-07"


def test_resolved_subject_set_for_build_rejects_incomplete_universe_policy(tmp_path):
    from alpha_os.subject_set_backfill_service import resolve_subject_set_for_build
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    _register_multi_subject_set(store, complete_universe_policy=False)

    try:
        with pytest.raises(ValueError, match="subject set universe policy is incomplete"):
            resolve_subject_set_for_build(
                store,
                SimpleNamespace(
                    subject_set_id="macro_pair",
                    subject_binding=[],
                    observation_spec=[],
                ),
            )
    finally:
        store.close()


def test_run_subject_set_backfill_rejects_incomplete_universe_policy(tmp_path):
    from alpha_os.subject_set_backfill_service import run_subject_set_backfill
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()
    _register_multi_subject_set(store, complete_universe_policy=False)
    subject_set_state = store.get_subject_set("macro_pair")
    assert subject_set_state is not None

    try:
        with pytest.raises(ValueError, match="subject set universe policy is incomplete"):
            run_subject_set_backfill(
                store,
                subject_set=subject_set_state.definition,
                subject_set_id="macro_pair",
                signal_spec_ids=[],
                target_id="residual_return_1d",
                start_date="2026-03-20",
                end_date="2026-03-25",
                base_url="http://example.com",
                pre_screen_top_k_per_kind=None,
                pre_screen_min_abs_corr=0.0,
            )
    finally:
        store.close()


def test_evaluation_report_contract_validation_reports_missing_structured_contracts():
    from alpha_os.cross_instrument_contract import CrossInstrumentReportContract
    from alpha_os.evaluation_report import EvaluationTaskResult, EvaluationReport
    from alpha_os.evaluation_report_contract_validation import validate_evaluation_report_contract

    validation_run = SimpleNamespace(
        run_id="validation:test",
        cross_instrument_contract=CrossInstrumentReportContract(
            contract_fields=("subject_set",),
            outcome_fields=("mean_net",),
        ),
        validation_result_set=None,
    )
    report = EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="evaluation_spec:test",
        task_results=(
            EvaluationTaskResult(
                evaluation_task_id="case:test",
                strategy_id="strategy:test",
            ),
        ),
        created_at="2026-04-18T00:00:00Z",
        cross_instrument_contract=CrossInstrumentReportContract(
            contract_fields=("strategy",),
            outcome_fields=("metric_group_outcomes",),
        ),
    )

    result = validate_evaluation_report_contract(
        validation_run=validation_run,
        evaluation_report=report,
    )

    assert result.passed is False
    assert "validation contract does not match the canonical evaluation report contract" in result.issues
    assert "validation result set is missing" in result.issues
    assert (
        "evaluation report contract does not match the canonical evaluation report contract"
        in result.issues
    )
    assert (
        "evaluation report task result case:test is missing strategy contract fields" in result.issues
    )
    assert (
        "evaluation report task result case:test is missing subject-set contract groups"
        in result.issues
    )
    assert "evaluation report task result case:test is missing constraint stages" in result.issues


def test_evaluation_report_contract_validation_requires_active_constraint_fields():
    from alpha_os.cross_instrument_contract import (
        default_evaluation_report_cross_instrument_contract,
        default_validation_result_set_cross_instrument_contract,
    )
    from alpha_os.evaluation_report import EvaluationTaskResult, EvaluationReport
    from alpha_os.evaluation_report_contract_validation import validate_evaluation_report_contract
    from alpha_os.validation_result_set import (
        ValidationDecisionSummary,
        ValidationResultSet,
    )

    validation_run = SimpleNamespace(
        run_id="validation:test",
        cross_instrument_contract=default_validation_result_set_cross_instrument_contract(),
        validation_result_set=ValidationResultSet(
            signal_summaries=(
                SimpleNamespace(
                    signal_id="momentum_1d",
                    conditions=1,
                    positive_corr=1,
                    mean_corr=0.1,
                    mean_mmc=None,
                ),
            ),
            meta_summaries=(
                SimpleNamespace(
                    aggregation_kind="active_equal_mean", conditions=1, wins=1, mean_corr=0.1
                ),
            ),
            decision_summaries=(
                ValidationDecisionSummary(
                    subject_set_id="core_crypto",
                    aggregation_kind="active_equal_mean",
                    conditions=1,
                    wins=1,
                    negative_conditions=0,
                    mean_net=0.1,
                    worst_net=0.1,
                    mean_drawdown=0.02,
                    mean_gross_notional=0.5,
                    mean_net_notional=0.0,
                    mean_long_notional=0.25,
                    mean_short_notional=0.25,
                    mean_traded_notional=0.1,
                    total_cost_notional=0.01,
                    total_funding_cost_notional=0.004,
                    total_borrow_cost_notional=0.003,
                    total_roll_cost_notional=0.002,
                    subject_set_contract_groups=(
                        "instrument",
                        "observation_spec",
                        "binding",
                        "universe_policy",
                    ),
                    universe_policy_fields={
                        "base_currency": "USD",
                        "trading_calendar": "24x7",
                        "benchmark_id": "core_crypto",
                    },
                ),
            ),
        ),
    )
    report = EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="evaluation_spec:test",
        task_results=(
            EvaluationTaskResult(
                evaluation_task_id="case:test",
                strategy_id="strategy:test",
                strategy_contract_fields={
                    "subject_set": "core_crypto",
                    "selection": "top_k",
                    "sizing": "equal_weight",
                    "rebalance": "every_1_steps",
                    "long_only": "true",
                    "gross_exposure_cap": 1.0,
                    "gross_leverage_cap": 1.5,
                    "net_exposure_target": 0.3,
                    "base_currency": "USD",
                    "trading_calendar": "24x7",
                    "benchmark_id": "core_crypto",
                },
                subject_set_facts="bindings=2 instruments=2",
                subject_set_contract_groups=(
                    "instrument",
                    "observation_spec",
                    "binding",
                    "universe_policy",
                ),
                universe_policy_fields={
                    "base_currency": "USD",
                    "trading_calendar": "24x7",
                    "benchmark_id": "core_crypto",
                },
                constraint_stages=(
                    "sizing_time:target_vol",
                    "post_sizing_normalization:long_only,gross_exposure_cap,gross_leverage_cap,net_exposure_target",
                ),
            ),
        ),
        created_at="2026-04-18T00:00:00Z",
        cross_instrument_contract=default_evaluation_report_cross_instrument_contract(),
    )

    result = validate_evaluation_report_contract(
        validation_run=validation_run,
        evaluation_report=report,
    )

    assert result.passed is False
    assert (
        "evaluation report task result case:test strategy contract is missing active constraint field target_vol"
        in result.issues
    )
