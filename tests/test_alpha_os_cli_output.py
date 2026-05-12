from __future__ import annotations

import pytest

from alpha_os.cli_output import (
    print_evaluation_tasks,
    print_evaluation_report,
    print_evaluation_snapshot,
    print_subject_sets,
    print_validation_result_set,
)
from alpha_os.evaluation_task import EvaluationTask
from alpha_os.portfolio_construction_config import (
    PortfolioConstructionSizingSpec,
    PortfolioConstructionSpec,
)
from alpha_os.store import EvaluationSnapshot
from alpha_os.portfolio_decision import (
    InstrumentSpec,
    ObservationSpec,
    SubjectObservationBinding,
    SubjectSet,
    UniversePolicySpec,
)
from alpha_os.trading_strategy import (
    AdaptationPolicySpec,
    ExecutionPolicySpec,
    HoldingCostPolicySpec,
    RebalanceFrictionPolicySpec,
    StrategyPortfolioSpec,
    TradingStrategyScopeSpec,
    TradingStrategySpec,
)


def _strategy_portfolio(
    *,
    selection_kind: str,
    sizing_method: str,
    direction_mode: str | None,
    gross_exposure_cap: float | None,
    top_k: int | None = None,
    target_vol: float | None = None,
    gross_leverage_cap: float | None = None,
    net_exposure_target: float | None = None,
    asset_class_weight_caps: dict[str, float] | None = None,
    cluster_weight_caps: dict[str, float] | None = None,
    rebalance_friction_policy: RebalanceFrictionPolicySpec | None = None,
    execution_policy: ExecutionPolicySpec | None = None,
    holding_cost_policy: HoldingCostPolicySpec | None = None,
) -> StrategyPortfolioSpec:
    return StrategyPortfolioSpec(
        portfolio_construction=PortfolioConstructionSpec(
            sizing_policy=PortfolioConstructionSizingSpec(
                sizing_method=sizing_method,
            ),
            direction_mode=direction_mode,
            gross_exposure_cap=gross_exposure_cap,
            target_vol=target_vol,
            gross_leverage_cap=gross_leverage_cap,
            net_exposure_target=net_exposure_target,
            asset_class_weight_caps=(
                {} if asset_class_weight_caps is None else dict(asset_class_weight_caps)
            ),
            cluster_weight_caps=(
                {} if cluster_weight_caps is None else dict(cluster_weight_caps)
            ),
        ),
        rebalance_friction_policy=(
            RebalanceFrictionPolicySpec()
            if rebalance_friction_policy is None
            else rebalance_friction_policy
        ),
        execution_policy=ExecutionPolicySpec() if execution_policy is None else execution_policy,
        holding_cost_policy=(
            HoldingCostPolicySpec()
            if holding_cost_policy is None
            else holding_cost_policy
        ),
        selection_kind=selection_kind,
        top_k=top_k,
    )


def test_print_evaluation_tasks_includes_execution_and_holding_costs(capsys):
    case = EvaluationTask(
        evaluation_task_id="case:test",
        strategy_id="strategy:test",
        evaluation_spec_id="protocol:test",
    )

    print_evaluation_tasks([case])
    captured = capsys.readouterr().out

    assert "case:test" in captured
    assert "strategy=strategy:test" in captured
    assert "signal_train=" not in captured
    assert "gross_exposure_cap=1.0" not in captured
    assert "market_impact_bps=5.0" not in captured
    assert "borrow_fee_bps_per_step=2.5" not in captured


def test_print_evaluation_snapshot_includes_replay_artifacts(capsys):
    snapshot = EvaluationSnapshot(
        evaluation_id="BTC:residual_return_3d:2026-04-17",
        subject_id="BTC_spot",
        asset="BTC",
        target_id="residual_return_3d",
        signal_id="time_series_trend__daily_close__lookback_20@BTC_spot",
        prediction_value=0.12,
        observation_value=-0.03,
        signed_edge=-0.15,
        absolute_error=0.15,
        input_source="signal_noise_backfill",
        input_range_start="2026-04-10",
        input_range_end="2026-04-17",
        funding_cost_bps=1.5,
        borrow_fee_bps=2.5,
        roll_cost_bps=0.75,
        contract_multiplier=5.0,
        observation_spec_id="macro_close",
        observable_id="daily_close",
        adapter_kind="signal_noise_asset_observable",
        created_at="2026-04-17T00:00:00Z",
    )

    print_evaluation_snapshot(snapshot, created=True)
    captured = capsys.readouterr().out

    assert "Evaluation [created] BTC:residual_return_3d:2026-04-17" in captured
    assert "Replay:   funding_bps=1.500000 borrow_bps=2.500000 roll_bps=0.750000 multiplier=5.000000" in captured


def test_print_evaluation_report_includes_subject_set_context(capsys):
    from alpha_os.evaluation_report import EvaluationTaskResult, EvaluationReport
    from alpha_os.strategy_sleeves import SleeveAttributionSummary

    report = EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="protocol:test",
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
                    "direction_mode": "long_only",
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
                        "post_sizing_normalization:direction_mode,gross_exposure_cap,"
                        "gross_leverage_cap,net_exposure_target"
                    ),
                    "funding_bps_per_step": 1.5,
                    "borrow_fee_bps_per_step": 2.5,
                    "subject_set": "global_macro_core",
                    "base_currency": "USD",
                    "trading_calendar": "24x7",
                    "benchmark_id": "global_macro_core",
                },
                subject_set_facts=(
                    "bindings=2 instruments=2 subject_kinds=future,perp "
                    "instrument_types=future,perp "
                    "contract_groups=instrument,observation_spec,binding,universe_policy"
                ),
                subject_set_contract_groups=(
                    "instrument",
                    "observation_spec",
                    "binding",
                    "universe_policy",
                ),
                universe_policy_fields={
                    "base_currency": "USD",
                    "trading_calendar": "24x7",
                    "benchmark_id": "global_macro_core",
                },
                constraint_stages=(
                    "sizing_time:target_vol",
                    "post_sizing_normalization:direction_mode,gross_exposure_cap,gross_leverage_cap,net_exposure_target",
                ),
                sleeve_attribution_summaries=(
                    SleeveAttributionSummary(
                        sleeve_id="trend_core",
                        sleeve_kind="trend",
                        risk_budget=1.0,
                        subject_count=2,
                        mean_signal=0.25,
                        mean_abs_signal=0.5,
                        mean_gross_notional_exposure=1.2,
                        mean_net_notional_exposure=0.1,
                        mean_long_notional_exposure=0.65,
                        mean_short_notional_exposure=0.55,
                        total_cost_notional=0.03,
                        total_funding_cost_notional=0.01,
                        total_borrow_cost_notional=0.015,
                        total_roll_cost_notional=0.005,
                    ),
                ),
            ),
        ),
        created_at="2026-04-18T00:00:00Z",
    )

    print_evaluation_report(report)
    captured = capsys.readouterr().out

    assert "strategy=strategy:test" in captured
    assert "CrossInstrumentReportContract:" in captured
    assert (
        "contract=strategy,subject_set,universe_policy,instrument_mix,selection,sizing,rebalance,risk_caps,costs "
        "outcomes=metric_group_outcomes,failure_finding_outcomes"
    ) in captured
    assert (
        "ReportUnits: task_result=strategy_id+evaluation_task_id, "
        "metric_group_outcome=evaluation_task_id+metric_group_name+source, "
        "failure_finding_outcome=evaluation_task_id+metric_group_name+source"
    ) in captured
    assert "MetricContracts:" in captured
    assert "TaskResultDetails:" in captured
    assert (
        "metric_group_outcome:decision_quality="
        "mean_decision_net_return+best_decision_net_return+mean_decision_drawdown+mean_decision_turnover+"
        "mean_decision_gross_leverage_exposure"
    ) in captured
    assert "failure_finding_outcome:decision_quality=finding_count+max_severity" in captured
    assert "signal_discovery=signal-discovery:test" in captured
    assert "construction=active_portfolio" in captured
    assert "selection=top_k" in captured
    assert "top_k=3" in captured
    assert "sizing=equal_weight" in captured
    assert "rebalance=every_1_steps" in captured
    assert "direction_mode=long_only" in captured
    assert "gross_exposure_cap=1.0" in captured
    assert "target_vol=0.12" in captured
    assert "gross_leverage_cap=1.5" in captured
    assert "net_exposure_target=0.3" in captured
    assert "subject_set=global_macro_core" in captured
    assert "base_currency=USD" in captured
    assert "trading_calendar=24x7" in captured
    assert "benchmark_id=global_macro_core" in captured
    assert "subject_set_contract_groups=instrument,observation_spec,binding,universe_policy" in captured
    assert "universe_policy=base_currency=USD trading_calendar=24x7 benchmark_id=global_macro_core" in captured
    assert "constraint_stages=sizing_time:target_vol;" in captured
    assert "post_sizing_normalization:direction_mode,gross_exposure_cap,gross_leverage_cap,net_exposure_target" in captured
    assert "summary=[bindings=2 instruments=2 subject_kinds=future,perp instrument_types=future,perp" in captured
    assert "contract_groups=instrument,observation_spec,binding,universe_policy" in captured
    assert "sleeve=trend_core kind=trend risk_budget=1.000000 subjects=2" in captured
    assert "gross=1.200000 net=0.100000 long=0.650000 short=0.550000" in captured
    assert "cost=0.030000 funding=0.010000 borrow=0.015000 roll=0.005000" in captured
    assert "findings=0 max_severity=0.000000" not in captured


def test_print_evaluation_report_lists_task_result_details(capsys):
    from alpha_os.evaluation_report import EvaluationTaskResult, EvaluationReport
    from alpha_os.evaluation_report import EvaluationMetricGroupResult

    report = EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="protocol:test",
        task_results=(
            EvaluationTaskResult(
                evaluation_task_id="case:primary",
                strategy_id="strategy:primary",
                construction_kind="hold_baseline",
                strategy_contract_fields={
                    "construction_kind": "hold_baseline",
                    "holding_style": "equal_weight_hold",
                    "selection": "all_assets",
                    "sizing": "equal_weight",
                    "rebalance": "every_252_steps",
                    "direction_mode": "long_only",
                },
                metric_group_results=(
                    EvaluationMetricGroupResult(
                        metric_group_name="decision_quality",
                        source="native",
                        metrics={
                            "mean_decision_net_return": 0.04,
                            "mean_decision_drawdown": 0.05,
                            "mean_decision_turnover": 0.004,
                        },
                    ),
                    EvaluationMetricGroupResult(
                        metric_group_name="cost_drag",
                        source="native",
                        metrics={"execution_cost_to_gross_pnl": 0.10},
                    ),
                    EvaluationMetricGroupResult(
                        metric_group_name="portfolio_concentration",
                        source="native",
                        metrics={"mean_top3_gross_share": 0.60},
                    ),
                    EvaluationMetricGroupResult(
                        metric_group_name="execution_trace",
                        source="native",
                        metrics={"utility_rejected_turnover": 0.0},
                    ),
                    EvaluationMetricGroupResult(
                        metric_group_name="portfolio_target_return_alignment",
                        source="native",
                        metrics={"mean_range_portfolio_target_return_corr": 0.03},
                    ),
                ),
            ),
            EvaluationTaskResult(
                evaluation_task_id="case:candidate",
                strategy_id="strategy:candidate",
                construction_kind="active_portfolio",
                strategy_contract_fields={
                    "construction_kind": "active_portfolio",
                    "selection": "all_assets",
                    "sizing": "diversified_risk_budget",
                    "rebalance": "every_1_steps",
                },
                metric_group_results=(
                    EvaluationMetricGroupResult(
                        metric_group_name="decision_quality",
                        source="native",
                        metrics={
                            "mean_decision_net_return": 0.02,
                            "mean_decision_drawdown": 0.04,
                            "mean_decision_turnover": 0.01,
                        },
                    ),
                    EvaluationMetricGroupResult(
                        metric_group_name="cost_drag",
                        source="native",
                        metrics={"execution_cost_to_gross_pnl": 0.25},
                    ),
                    EvaluationMetricGroupResult(
                        metric_group_name="portfolio_concentration",
                        source="native",
                        metrics={"mean_top3_gross_share": 0.50},
                    ),
                    EvaluationMetricGroupResult(
                        metric_group_name="execution_trace",
                        source="native",
                        metrics={"utility_rejected_turnover": 0.02},
                    ),
                    EvaluationMetricGroupResult(
                        metric_group_name="portfolio_target_return_alignment",
                        source="native",
                        metrics={"mean_range_portfolio_target_return_corr": 0.07},
                    ),
                ),
            ),
            EvaluationTaskResult(
                evaluation_task_id="case:diagnostic",
                strategy_id="strategy:diagnostic",
                construction_kind="active_portfolio",
                strategy_contract_fields={
                    "construction_kind": "active_portfolio",
                    "selection": "all_assets",
                    "sizing": "diversified_risk_budget",
                    "rebalance": "every_1_steps",
                },
            ),
        ),
        created_at="2026-04-18T00:00:00Z",
    )

    print_evaluation_report(report)
    captured = capsys.readouterr().out

    assert "TaskResultDetails:" in captured
    assert "PrimaryComparison:" not in captured
    assert "vs_baseline" not in captured
    assert "CaseMetricFacts:" in captured
    assert (
        "Task: case:primary "
        "net=0.040000 drawdown=0.050000 turnover=0.004000 "
        "cost_drag=0.100000 top3_share=0.600000 "
        "target_return_corr=0.030000 utility_rejected_turnover=0.000000"
    ) in captured
    assert (
        "Task: case:candidate "
        "net=0.020000 drawdown=0.040000 turnover=0.010000 "
        "cost_drag=0.250000 top3_share=0.500000 "
        "target_return_corr=0.070000 utility_rejected_turnover=0.020000"
    ) in captured
    assert (
        "Task: case:diagnostic "
        "net=- drawdown=- turnover=- cost_drag=- top3_share=- "
        "target_return_corr=- utility_rejected_turnover=-"
    ) in captured
    assert (
        "Task: case:primary "
        "construction=hold_baseline"
    ) in captured
    assert (
        "Task: case:candidate "
        "construction=active_portfolio"
    ) in captured
    assert (
        "Task: case:diagnostic "
        "construction=active_portfolio"
    ) in captured


def test_evaluation_task_result_builds_cross_instrument_outcome():
    from alpha_os.evaluation_report import EvaluationTaskResult, EvaluationMetricGroupResult, EvaluationFailureFinding, EvaluationFailureFindingGroup

    task_result = EvaluationTaskResult(
        evaluation_task_id="case:test",
        strategy_id="strategy:test",
        metric_group_results=(
            EvaluationMetricGroupResult(
                metric_group_name="decision_quality",
                source="native_plan",
                metrics={"mean_decision_net_return": 0.12},
            ),
        ),
        failure_finding_groups=(
            EvaluationFailureFindingGroup(
                metric_group_name="decision_quality",
                source="native_plan",
                findings=(
                    EvaluationFailureFinding(
                        label="tail",
                        severity=0.25,
                        metrics={"decision_net_return": -0.25},
                    ),
                ),
            ),
        ),
    )

    assert task_result.cross_instrument_outcome is not None
    assert task_result.cross_instrument_outcome.metric_group_outcomes[0].metric_group_name == "decision_quality"
    assert (
        task_result.cross_instrument_outcome.metric_group_outcomes[0].metrics["mean_decision_net_return"]
        == 0.12
    )
    assert task_result.cross_instrument_outcome.failure_finding_outcomes[0].metric_group_name == "decision_quality"
    assert task_result.cross_instrument_outcome.failure_finding_outcomes[0].finding_count == 1
    assert task_result.cross_instrument_outcome.failure_finding_outcomes[0].max_severity == 0.25


def test_evaluation_task_result_rejects_legacy_profiles_field():
    from alpha_os.evaluation_report import EvaluationTaskResult

    with pytest.raises(ValueError, match="profiles field is no longer supported"):
        EvaluationTaskResult.from_document(
            {
                "evaluation_task_id": "case:test",
                "strategy_id": "strategy:test",
                "profiles": [],
                "failure_finding_groups": [],
            }
        )


def test_evaluation_task_result_rejects_legacy_failure_profiles_field():
    from alpha_os.evaluation_report import EvaluationTaskResult

    with pytest.raises(ValueError, match="failure_profiles field is no longer supported"):
        EvaluationTaskResult.from_document(
            {
                "evaluation_task_id": "case:test",
                "strategy_id": "strategy:test",
                "metric_group_results": [],
                "failure_profiles": [],
            }
        )


def test_evaluation_task_result_rejects_legacy_subject_set_summary_field():
    from alpha_os.evaluation_report import EvaluationTaskResult

    with pytest.raises(
        ValueError,
        match="subject_set_summary field is no longer supported",
    ):
        EvaluationTaskResult.from_document(
            {
                "evaluation_task_id": "case:test",
                "strategy_id": "strategy:test",
                "metric_group_results": [],
                "failure_finding_groups": [],
                "subject_set_summary": "bindings=2 instruments=2",
            }
        )


def test_resolve_report_strategy_context_includes_subject_set_facts(tmp_path):
    from alpha_os.evaluation_report import EvaluationTaskResult, EvaluationReport
    from alpha_os.evaluation_report_service import resolve_report_strategy_context
    from alpha_os.store import EvaluationStore

    store = EvaluationStore(tmp_path / "runtime.db")
    store.ensure_schema()
    store.upsert_subject_set(
        "global_macro_core",
        definition=SubjectSet(
            subject_set_id="global_macro_core",
            instruments=(
                InstrumentSpec(
                    instrument_id="es_fut",
                    instrument_type="future",
                    asset="ES",
                    asset_class="equity_index",
                    region="us",
                    cluster="risk",
                ),
                InstrumentSpec(
                    instrument_id="btc_perp",
                    instrument_type="perp",
                    asset="BTC",
                    asset_class="crypto",
                    region="global",
                    cluster="alt",
                ),
            ),
            observation_specs=(
                ObservationSpec(
                    observation_spec_id="es_close",
                    observable_id="daily_close",
                ),
                ObservationSpec(
                    observation_spec_id="btc_close",
                    observable_id="daily_close",
                ),
            ),
            bindings=(
                SubjectObservationBinding(
                    subject_id="ES_front",
                    asset="ES",
                    observation_spec_id="es_close",
                    subject_kind="future",
                    instrument_id="es_fut",
                ),
                SubjectObservationBinding(
                    subject_id="BTC_perp",
                    asset="BTC",
                    observation_spec_id="btc_close",
                    subject_kind="perp",
                    instrument_id="btc_perp",
                ),
            ),
            universe_policy=UniversePolicySpec(
                base_currency="USD",
                trading_calendar="24x7",
                benchmark_id="global_macro_core",
            ),
        ),
    )
    store.upsert_trading_strategy(
        trading_strategy=TradingStrategySpec(
            strategy_id="strategy:test",
            label="Test Strategy",
            scope=TradingStrategyScopeSpec(
                subject_set_id="global_macro_core",
                target_id="residual_return_3d",
            ),
            signal_discovery_id="signal-discovery:test",
            position_rule_id="constant_hold",
            family_mix=None,
            portfolio=_strategy_portfolio(
                selection_kind="top_k",
                top_k=3,
                sizing_method="equal_weight",
                direction_mode="long_only",
                gross_exposure_cap=1.0,
                target_vol=0.12,
                gross_leverage_cap=1.5,
                net_exposure_target=0.3,
                asset_class_weight_caps={"equity_index": 0.6},
                cluster_weight_caps={"risk": 0.4},
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
            ),
            created_at="2026-04-18T00:00:00Z",
            adaptation_policy=AdaptationPolicySpec(enabled=True, adaptation_blend=0.2),
        )
    )
    report = EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="protocol:test",
        task_results=(
            EvaluationTaskResult(
                evaluation_task_id="case:test",
                strategy_id="strategy:test",
                signal_discovery_id="signal-discovery:test",
            ),
        ),
        created_at="2026-04-18T00:00:00Z",
    )

    contexts = resolve_report_strategy_context(store, report_state=report)

    context = contexts["strategy:test"]
    assert "selection=top_k" in context
    assert "subject_set=global_macro_core" in context
    assert "base_currency=USD" in context
    assert "trading_calendar=24x7" in context
    assert "benchmark_id=global_macro_core" in context
    assert "target_vol=0.12" in context
    assert "gross_leverage_cap=1.5" in context
    assert "net_exposure_target=0.3" in context
    assert "asset_class_weight_caps=equity_index=0.6" in context
    assert "cluster_weight_caps=risk=0.4" in context
    assert "constraint_stages=sizing_time:target_vol;" in context
    assert (
        "post_sizing_normalization:direction_mode,gross_exposure_cap,gross_leverage_cap,"
        "net_exposure_target,asset_class_weight_caps,cluster_weight_caps"
    ) in context
    assert "summary=[bindings=2 instruments=2" in context
    assert "subject_kinds=future,perp" in context
    assert "instrument_types=future,perp" in context
    assert "asset_classes=crypto,equity_index" in context
    assert "regions=global,us" in context
    assert "clusters=alt,risk" in context
    assert "contract_groups=instrument,observation_spec,binding,universe_policy" in context
    assert "universe_policy=[base_currency=USD trading_calendar=24x7 benchmark_id=global_macro_core]" in context
    assert "base_currency=USD" in context
    assert "trading_calendar=24x7" in context
    assert "benchmark_id=global_macro_core" in context


def test_resolve_report_strategy_context_rejects_incomplete_universe_policy(tmp_path):
    from alpha_os.evaluation_report import EvaluationTaskResult, EvaluationReport
    from alpha_os.evaluation_report_service import resolve_report_strategy_context
    from alpha_os.store import EvaluationStore

    store = EvaluationStore(tmp_path / "runtime.db")
    store.ensure_schema()
    store.upsert_subject_set(
        "macro_pair",
        definition=SubjectSet(
            subject_set_id="macro_pair",
            instruments=(
                InstrumentSpec(
                    instrument_id="btc_spot",
                    instrument_type="spot",
                    asset="BTC",
                ),
                InstrumentSpec(
                    instrument_id="eth_spot",
                    instrument_type="spot",
                    asset="ETH",
                ),
            ),
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
                    instrument_id="btc_spot",
                ),
                SubjectObservationBinding(
                    subject_id="ETH_spot",
                    asset="ETH",
                    observation_spec_id="eth_close",
                    instrument_id="eth_spot",
                ),
            ),
            universe_policy=UniversePolicySpec(
                base_currency="USD",
                trading_calendar=None,
                benchmark_id=None,
            ),
        ),
    )
    store.upsert_trading_strategy(
        trading_strategy=TradingStrategySpec(
            strategy_id="strategy:invalid",
            label="Invalid Strategy",
            scope=TradingStrategyScopeSpec(
                subject_set_id="macro_pair",
                target_id="residual_return_3d",
            ),
            signal_discovery_id="signal-discovery:test",
            position_rule_id="constant_hold",
            family_mix=None,
            portfolio=_strategy_portfolio(
                selection_kind="top_k",
                top_k=2,
                sizing_method="equal_weight",
                direction_mode="long_only",
                gross_exposure_cap=1.0,
                rebalance_friction_policy=RebalanceFrictionPolicySpec(
                    turnover_friction=0.0,
                    no_trade_band=0.0,
                ),
                execution_policy=ExecutionPolicySpec(
                    market_impact_bps=0.0,
                    fee_bps=0.0,
                    bid_ask_spread_bps=0.0,
                ),
                holding_cost_policy=HoldingCostPolicySpec(
                    funding_bps_per_step=0.0,
                    borrow_fee_bps_per_step=0.0,
                ),
            ),
            created_at="2026-04-18T00:00:00Z",
        )
    )
    report = EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="protocol:test",
        task_results=(
            EvaluationTaskResult(
                evaluation_task_id="case:test",
                strategy_id="strategy:invalid",
            ),
        ),
        created_at="2026-04-18T00:00:00Z",
    )

    with pytest.raises(ValueError, match="subject set universe policy is incomplete"):
        resolve_report_strategy_context(store, report_state=report)


def test_current_evaluation_task_metadata_enriches_legacy_report(tmp_path):
    from alpha_os.cli import _with_current_evaluation_task_metadata
    from alpha_os.evaluation_report import EvaluationTaskResult, EvaluationReport
    from alpha_os.store import EvaluationStore

    store = EvaluationStore(tmp_path / "runtime.db")
    store.ensure_schema()
    store.upsert_evaluation_task(
        task=EvaluationTask(
            evaluation_task_id="case:hold",
            strategy_id="strategy:hold",
            evaluation_spec_id="protocol:test",
        )
    )
    legacy_report = EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="protocol:test",
        task_results=(
            EvaluationTaskResult(
                evaluation_task_id="case:hold",
                strategy_id="strategy:hold",
                strategy_contract_fields={
                    "construction_kind": "active_portfolio",
                    "selection": "all_assets",
                    "sizing": "equal_weight",
                    "active_overlay": "rank_tilt",
                    "sizing_family": "risk_budget_allocator",
                    "subject_set": "global_macro_core",
                    "base_currency": "USD",
                },
            ),
        ),
        created_at="2026-04-23T00:00:00Z",
    )

    enriched_report = _with_current_evaluation_task_metadata(store, legacy_report)

    task_result = enriched_report.task_results[0]
    assert task_result.construction_kind == "active_portfolio"
    assert task_result.strategy_contract_fields["construction_kind"] == "active_portfolio"
    assert task_result.strategy_contract_fields["active_overlay"] == "rank_tilt"
    assert task_result.strategy_contract_fields["sizing_family"] == "risk_budget_allocator"
    assert task_result.strategy_contract_fields["subject_set"] == "global_macro_core"
    assert task_result.strategy_contract_fields["base_currency"] == "USD"


def test_evaluation_report_roundtrips_cross_instrument_outcome():
    from alpha_os.evaluation_report import (
        EvaluationTaskResult,
        EvaluationMetricGroupResult,
        EvaluationReport,
    )

    report = EvaluationReport(
        evaluation_report_id="report:test",
        evaluation_spec_id="protocol:test",
        task_results=(
            EvaluationTaskResult(
                evaluation_task_id="case:test",
                strategy_id="strategy:test",
                strategy_contract_fields={"selection": "top_k"},
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
                    "benchmark_id": "global_macro_core",
                },
                constraint_stages=(
                    "sizing_time:target_vol",
                    "post_sizing_normalization:direction_mode,gross_exposure_cap,gross_leverage_cap,net_exposure_target",
                ),
                metric_group_results=(
                    EvaluationMetricGroupResult(
                        metric_group_name="decision_quality",
                        source="native_plan",
                        metrics={"mean_decision_net_return": 0.12},
                    ),
                ),
            ),
        ),
        created_at="2026-04-18T00:00:00Z",
    )

    restored = EvaluationReport.from_document(
        evaluation_report_id="report:test",
        document=report.to_document(),
    )

    assert restored.cross_instrument_contract.outcome_fields == (
        "metric_group_outcomes",
        "failure_finding_outcomes",
    )
    assert tuple(
        item.unit_id for item in restored.cross_instrument_contract.report_units
    ) == (
        "task_result",
        "metric_group_outcome",
        "failure_finding_outcome",
    )
    assert any(
        item.outcome_kind == "metric_group_outcome"
        and item.metric_group_name == "decision_quality"
        for item in restored.cross_instrument_contract.metric_contracts
    )
    assert "task_results" in restored.to_document()
    assert "summaries" not in restored.to_document()
    task_result = restored.task_results[0]
    assert "metric_group_results" in task_result.to_document()
    assert "profiles" not in task_result.to_document()
    assert task_result.cross_instrument_outcome is not None
    assert task_result.strategy_contract_fields == {"selection": "top_k"}
    assert task_result.subject_set_facts == "bindings=2 instruments=2"
    assert task_result.cross_instrument_outcome.metric_group_outcomes[0].metric_group_name == "decision_quality"
    assert task_result.subject_set_contract_groups == (
        "instrument",
        "observation_spec",
        "binding",
        "universe_policy",
    )
    assert task_result.universe_policy_fields == {
        "base_currency": "USD",
        "trading_calendar": "24x7",
        "benchmark_id": "global_macro_core",
    }
    assert task_result.constraint_stages == (
        "sizing_time:target_vol",
        "post_sizing_normalization:direction_mode,gross_exposure_cap,gross_leverage_cap,net_exposure_target",
    )


def test_print_validation_result_set_includes_subject_set_facts(capsys):
    from types import SimpleNamespace
    from alpha_os.validation_result_set import build_validation_result_set

    run = SimpleNamespace(run_id="validation:test")
    decision_result = SimpleNamespace(
        date_range_label="recent",
        target_id="residual_return_3d",
        subject_set_id="global_macro_core",
        window_size=20,
        aggregation_kind="active_equal_mean",
        net_return_total=0.12,
        max_drawdown=0.03,
        mean_traded_notional=0.4,
        mean_gross_notional_exposure=0.8,
        mean_net_notional_exposure=0.1,
        mean_long_notional_exposure=0.45,
        mean_short_notional_exposure=0.35,
        cost_notional_total=0.01,
        funding_cost_notional_total=0.004,
        borrow_cost_notional_total=0.003,
        roll_cost_notional_total=0.002,
    )
    run.validation_result_set = build_validation_result_set(
        [],
        [],
        [decision_result],
        subject_set_contract_groups_by_id={
            "global_macro_core": (
                "instrument",
                "observation_spec",
                "binding",
                "universe_policy",
            ),
        },
        universe_policy_by_subject_set_id={
            "global_macro_core": {
                "base_currency": "USD",
                "trading_calendar": "24x7",
                "benchmark_id": "global_macro_core",
            },
        },
    )

    print_validation_result_set(
        run,
        [],
        [],
        [decision_result],
        subject_set_facts_by_id={
            "global_macro_core": (
                "bindings=2 instruments=2 subject_kinds=future,perp "
                "instrument_types=future,perp asset_classes=crypto,equity_index "
                "contract_groups=instrument,observation_spec,binding,universe_policy"
            )
        },
    )
    captured = capsys.readouterr().out

    assert "CrossInstrumentReportContract:" in captured
    assert "contract=subject_set,universe_policy,instrument_mix,aggregation_kind outcomes=mean_net,mean_drawdown,mean_net_notional,mean_long_notional,mean_short_notional,mean_traded_notional,total_cost_notional,total_funding_cost_notional,total_borrow_cost_notional,total_roll_cost_notional" in captured
    assert "ReportUnits: signal_level=signal_id, meta_aggregation=aggregation_kind, decision_aggregation=subject_set_id+aggregation_kind" in captured
    assert "subject_set=global_macro_core" in captured
    assert "subject_set_contract_groups=instrument,observation_spec,binding,universe_policy" in captured
    assert "universe_policy=base_currency=USD trading_calendar=24x7 benchmark_id=global_macro_core" in captured
    assert "summary=[bindings=2 instruments=2 subject_kinds=future,perp instrument_types=future,perp asset_classes=crypto,equity_index contract_groups=instrument,observation_spec,binding,universe_policy]" in captured


def test_print_subject_sets_includes_cross_asset_summary(capsys):
    subject_set = SubjectSet(
        subject_set_id="global_macro_core",
        instruments=(
            InstrumentSpec(
                instrument_id="ES_future",
                instrument_type="future",
                asset="ES",
                asset_class="equity_index",
                region="us",
                cluster="eq_us",
            ),
            InstrumentSpec(
                instrument_id="BTCUSDT_perp",
                instrument_type="perp",
                asset="BTCUSDT",
                venue="binance",
                asset_class="crypto",
                region="global",
                cluster="crypto_major",
            ),
        ),
        observation_specs=(
            ObservationSpec(
                observation_spec_id="macro_close",
                observable_id="daily_close",
                provided_observable_ids=("funding_rate", "basis"),
            ),
        ),
        bindings=(
            SubjectObservationBinding(
                subject_id="ES_future",
                subject_kind="future",
                asset="ES",
                observation_spec_id="macro_close",
                instrument_id="ES_future",
            ),
            SubjectObservationBinding(
                subject_id="BTCUSDT_perp",
                subject_kind="perp",
                asset="BTCUSDT",
                observation_spec_id="macro_close",
                instrument_id="BTCUSDT_perp",
            ),
        ),
        universe_policy=UniversePolicySpec(
            base_currency="USD",
            trading_calendar="24x7",
            benchmark_id="global_macro_core",
        ),
    )

    class State:
        subject_set_id = "global_macro_core"
        definition = subject_set

    print_subject_sets([State()])
    captured = capsys.readouterr().out

    assert "summary=[bindings=2 instruments=2" in captured
    assert "subject_kinds=future,perp" in captured
    assert "instrument_types=future,perp" in captured
    assert "asset_classes=crypto,equity_index" in captured
    assert "regions=global,us" in captured
    assert "clusters=crypto_major,eq_us" in captured
    assert "base_currency=USD" in captured
    assert "trading_calendar=24x7" in captured
    assert "benchmark_id=global_macro_core" in captured
    assert "contract_groups=instrument,observation_spec,binding,universe_policy" in captured


def test_subject_set_store_roundtrip_preserves_universe_policy(tmp_path):
    from alpha_os.store import EvaluationStore

    store = EvaluationStore(tmp_path / "runtime.db")
    store.ensure_schema()
    store.upsert_subject_set(
        "global_macro_core",
        definition=SubjectSet(
            subject_set_id="global_macro_core",
            observation_specs=(
                ObservationSpec(
                    observation_spec_id="macro_close",
                    observable_id="daily_close",
                ),
            ),
            bindings=(
                SubjectObservationBinding(
                    subject_id="BTC_spot",
                    subject_kind="asset",
                    asset="BTC",
                    observation_spec_id="macro_close",
                ),
            ),
            universe_policy=UniversePolicySpec(
                base_currency="USD",
                trading_calendar="24x7",
                benchmark_id="global_macro_core",
            ),
        ),
    )

    restored = store.get_subject_set("global_macro_core")

    assert restored is not None
    assert restored.definition.universe_policy.base_currency == "USD"
    assert restored.definition.universe_policy.trading_calendar == "24x7"
    assert restored.definition.universe_policy.benchmark_id == "global_macro_core"
    store.close()


def test_strategy_and_portfolio_construction_roundtrip_preserve_broader_constraints():
    portfolio_construction = PortfolioConstructionSpec(
        sizing_policy=PortfolioConstructionSizingSpec(
            sizing_method="signal_weighted",
            sizing_engine="rule_based",
        ),
        gross_exposure_cap=1.0,
        target_vol=0.12,
        gross_leverage_cap=1.5,
        net_exposure_target=0.3,
    )
    restored_portfolio_construction = PortfolioConstructionSpec.from_document(
        portfolio_construction.to_document()
    )

    assert restored_portfolio_construction == portfolio_construction

    short_only_construction = PortfolioConstructionSpec(direction_mode="short_only")
    restored_short_only_construction = PortfolioConstructionSpec.from_document(
        short_only_construction.to_document()
    )

    assert restored_short_only_construction.direction_mode == "short_only"
    assert restored_short_only_construction.long_only is False


def test_signed_mean_variance_sizing_spec_defaults_to_signed_optimizer_family():
    from alpha_os.portfolio_construction_config import PortfolioConstructionSizingSpec

    sizing = PortfolioConstructionSizingSpec(sizing_method="signed_mean_variance")
    restored = PortfolioConstructionSizingSpec.from_document(sizing.to_document())

    assert sizing.sizing_engine == "optimizer"
    assert sizing.sizing_family == "signed_optimizer"
    assert restored == sizing


def test_conviction_adjusted_hrp_sizing_spec_defaults_to_risk_budget_family():
    from alpha_os.portfolio_construction_config import PortfolioConstructionSizingSpec

    sizing = PortfolioConstructionSizingSpec(
        sizing_method="conviction_adjusted_hierarchical_risk_parity"
    )
    restored = PortfolioConstructionSizingSpec.from_document(sizing.to_document())

    assert sizing.sizing_engine == "history_based"
    assert sizing.sizing_family == "risk_budget_allocator"
    assert restored == sizing


def test_sizing_spec_rejects_inconsistent_family():
    from alpha_os.portfolio_construction_config import PortfolioConstructionSizingSpec

    with pytest.raises(ValueError, match="sizing_family must match"):
        PortfolioConstructionSizingSpec(
            sizing_method="signed_mean_variance",
            sizing_family="risk_budget_allocator",
        )
