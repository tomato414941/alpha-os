from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_default_cross_instrument_contracts_are_stable():
    from alpha_os.cross_instrument_contract import (
        default_evaluation_report_cross_instrument_contract,
        default_validation_result_set_cross_instrument_contract,
    )

    report = default_evaluation_report_cross_instrument_contract()
    validation = default_validation_result_set_cross_instrument_contract()

    assert report.contract_fields == (
        "strategy",
        "subject_set",
        "universe_policy",
        "instrument_mix",
        "selection",
        "sizing",
        "rebalance",
        "risk_caps",
        "costs",
    )
    assert report.outcome_fields == (
        "metric_group_outcomes",
        "failure_finding_outcomes",
    )
    assert tuple(item.unit_id for item in report.report_units) == (
        "task_result",
        "metric_group_outcome",
        "failure_finding_outcome",
    )
    assert report.report_units[0].fields == ("strategy_id", "evaluation_task_id")
    assert report.report_units[1].fields == (
        "evaluation_task_id",
        "metric_group_name",
        "source",
    )
    assert report.report_units[2].fields == (
        "evaluation_task_id",
        "metric_group_name",
        "source",
    )
    report_metric_contracts = {
        (item.outcome_kind, item.metric_group_name): item.metric_fields
        for item in report.metric_contracts
    }
    assert report_metric_contracts[("metric_group_outcome", "decision_quality")] == (
        "mean_decision_net_return",
        "best_decision_net_return",
        "mean_decision_drawdown",
        "mean_decision_turnover",
        "mean_decision_gross_leverage_exposure",
        "mean_decision_net_leverage_exposure",
        "mean_decision_long_leverage_exposure",
        "mean_decision_short_leverage_exposure",
        "mean_decision_gross_notional_exposure",
        "mean_decision_net_notional_exposure",
        "mean_decision_long_notional_exposure",
        "mean_decision_short_notional_exposure",
        "mean_decision_traded_notional",
        "total_decision_cost_notional",
        "total_decision_funding_cost_notional",
        "total_decision_borrow_cost_notional",
        "total_decision_roll_cost_notional",
        "mean_decision_step_count",
        "total_decision_step_count",
        "mean_step_net_return",
        "step_net_return_std",
        "pooled_step_max_drawdown",
        "annualized_step_sharpe",
    )
    assert report_metric_contracts[("metric_group_outcome", "prediction_diagnostics")] == (
        "mean_signal_forward_corr",
        "mean_signal_hit_rate",
        "mean_long_short_forward_spread",
        "mean_long_bucket_return",
        "mean_short_bucket_return",
        "mean_prediction_coverage",
        "positive_group_fraction",
    )
    assert report_metric_contracts[("metric_group_outcome", "portfolio_risk_budget")] == (
        "risk_normalization_mode",
        "allow_releverage",
        "target_gross_exposure",
        "mean_gross_budget_utilization",
        "mean_gross_budget_error",
        "mean_decision_gross_leverage_exposure",
    )
    assert report_metric_contracts[
        ("metric_group_outcome", "portfolio_construction_trace")
    ] == (
        "risk_budget_stage_mean_gross_delta",
        "target_vol_stage_mean_gross_delta",
        "net_target_stage_mean_net_delta",
        "top_k_stage_mean_active_count_delta",
    )
    assert report_metric_contracts[("metric_group_outcome", "execution_trace")] == (
        "mean_desired_turnover",
        "mean_executed_turnover",
        "mean_turnover_suppression",
        "mean_skipped_trade_count",
        "mean_expected_execution_cost",
        "mean_trade_utility",
        "negative_utility_trade_count",
        "negative_utility_trade_fraction",
        "utility_rejected_turnover",
        "priority_filled_turnover",
        "partial_fill_count",
    )
    assert report_metric_contracts[("metric_group_outcome", "cost_drag")] == (
        "cost_to_gross_pnl",
        "execution_cost_to_gross_pnl",
        "total_execution_cost_notional",
        "top_cost_subjects",
        "top_cost_clusters",
    )
    assert report_metric_contracts[("metric_group_outcome", "signal_churn")] == (
        "mean_signal_abs_change",
        "mean_signal_sign_flip_rate",
        "mean_desired_weight_change",
    )
    assert report_metric_contracts[("failure_finding_outcome", "decision_quality")] == (
        "finding_count",
        "max_severity",
    )
    assert report_metric_contracts[("failure_finding_outcome", "portfolio_risk_budget")] == (
        "finding_count",
        "max_severity",
    )
    assert validation.contract_fields == (
        "subject_set",
        "universe_policy",
        "instrument_mix",
        "aggregation_kind",
    )
    assert validation.outcome_fields == (
        "mean_net",
        "mean_drawdown",
        "mean_net_notional",
        "mean_long_notional",
        "mean_short_notional",
        "mean_traded_notional",
        "total_cost_notional",
        "total_funding_cost_notional",
        "total_borrow_cost_notional",
        "total_roll_cost_notional",
    )
    assert tuple(item.unit_id for item in validation.report_units) == (
        "signal_level",
        "meta_aggregation",
        "decision_aggregation",
    )
    assert validation.report_units[0].fields == ("signal_id",)
    assert validation.report_units[1].fields == ("aggregation_kind",)
    assert validation.report_units[2].fields == ("subject_set_id", "aggregation_kind")


def test_cross_instrument_contract_roundtrip_and_format_summary():
    from alpha_os.cross_instrument_contract import (
        CrossInstrumentReportUnit,
        CrossInstrumentMetricContract,
        CrossInstrumentReportContract,
    )

    contract = CrossInstrumentReportContract(
        contract_fields=("subject_set", "instrument_mix", "aggregation_kind"),
        outcome_fields=("mean_net", "mean_drawdown"),
        report_units=(
            CrossInstrumentReportUnit(
                unit_id="decision_aggregation",
                fields=("subject_set_id", "aggregation_kind"),
            ),
        ),
        metric_contracts=(
            CrossInstrumentMetricContract(
                outcome_kind="metric_group_outcome",
                metric_group_name="decision_quality",
                metric_fields=("mean_decision_net_return", "mean_decision_drawdown"),
            ),
        ),
    )

    restored = CrossInstrumentReportContract.from_document(contract.to_document())

    assert restored == contract
    assert (
        restored.format_summary()
        == "contract=subject_set,instrument_mix,aggregation_kind outcomes=mean_net,mean_drawdown"
    )
    assert (
        restored.format_report_units()
        == "decision_aggregation=subject_set_id+aggregation_kind"
    )
    assert (
        restored.format_metric_contracts()
        == "metric_group_outcome:decision_quality=mean_decision_net_return+mean_decision_drawdown"
    )


def test_cross_instrument_contract_rejects_legacy_comparison_units():
    from alpha_os.cross_instrument_contract import CrossInstrumentReportContract

    with pytest.raises(ValueError, match="comparison_units field is no longer supported"):
        CrossInstrumentReportContract.from_document(
            {
                "contract_fields": [],
                "outcome_fields": [],
                "comparison_units": [],
                "metric_contracts": [],
            }
        )


def test_cross_instrument_metric_contract_rejects_legacy_dimension_name_field():
    from alpha_os.cross_instrument_contract import CrossInstrumentMetricContract

    with pytest.raises(ValueError, match="dimension_name field is no longer supported"):
        CrossInstrumentMetricContract.from_document(
            {
                "outcome_kind": "metric_group_outcome",
                "dimension_name": "decision_quality",
                "metric_fields": [],
            }
        )


def test_validation_result_set_groups_by_subject_set_and_aggregation_kind():
    from alpha_os.validation_result_set import build_validation_result_set

    decision_results = [
        SimpleNamespace(
            date_range_label="recent",
            target_id="residual_return_3d",
            subject_set_id="global_macro_core",
            window_size=20,
            aggregation_kind="active_equal_mean",
            net_return_total=0.10,
            max_drawdown=0.05,
            mean_gross_notional_exposure=0.80,
            mean_net_notional_exposure=0.10,
            mean_long_notional_exposure=0.45,
            mean_short_notional_exposure=0.35,
            mean_traded_notional=0.40,
            cost_notional_total=0.01,
            funding_cost_notional_total=0.004,
            borrow_cost_notional_total=0.003,
            roll_cost_notional_total=0.002,
        ),
        SimpleNamespace(
            date_range_label="recent",
            target_id="residual_return_3d",
            subject_set_id="global_macro_core",
            window_size=20,
            aggregation_kind="corr_weighted_mean",
            net_return_total=0.08,
            max_drawdown=0.04,
            mean_gross_notional_exposure=0.75,
            mean_net_notional_exposure=0.05,
            mean_long_notional_exposure=0.40,
            mean_short_notional_exposure=0.35,
            mean_traded_notional=0.35,
            cost_notional_total=0.02,
            funding_cost_notional_total=0.005,
            borrow_cost_notional_total=0.004,
            roll_cost_notional_total=0.003,
        ),
        SimpleNamespace(
            date_range_label="stress",
            target_id="residual_return_3d",
            subject_set_id="global_macro_core",
            window_size=20,
            aggregation_kind="active_equal_mean",
            net_return_total=0.04,
            max_drawdown=0.03,
            mean_gross_notional_exposure=0.90,
            mean_net_notional_exposure=0.20,
            mean_long_notional_exposure=0.55,
            mean_short_notional_exposure=0.35,
            mean_traded_notional=0.50,
            cost_notional_total=0.03,
            funding_cost_notional_total=0.006,
            borrow_cost_notional_total=0.005,
            roll_cost_notional_total=0.004,
        ),
        SimpleNamespace(
            date_range_label="stress",
            target_id="residual_return_3d",
            subject_set_id="global_macro_core",
            window_size=20,
            aggregation_kind="corr_weighted_mean",
            net_return_total=0.04,
            max_drawdown=0.02,
            mean_gross_notional_exposure=0.70,
            mean_net_notional_exposure=0.00,
            mean_long_notional_exposure=0.35,
            mean_short_notional_exposure=0.35,
            mean_traded_notional=0.30,
            cost_notional_total=0.01,
            funding_cost_notional_total=0.004,
            borrow_cost_notional_total=0.003,
            roll_cost_notional_total=0.002,
        ),
        SimpleNamespace(
            date_range_label="recent",
            target_id="residual_return_3d",
            subject_set_id="core_crypto",
            window_size=20,
            aggregation_kind="active_equal_mean",
            net_return_total=-0.02,
            max_drawdown=0.08,
            mean_gross_notional_exposure=0.60,
            mean_net_notional_exposure=-0.10,
            mean_long_notional_exposure=0.25,
            mean_short_notional_exposure=0.35,
            mean_traded_notional=0.20,
            cost_notional_total=0.01,
            funding_cost_notional_total=0.002,
            borrow_cost_notional_total=0.003,
            roll_cost_notional_total=0.001,
        ),
    ]

    result_set = build_validation_result_set(
        [],
        [],
        decision_results,
        subject_set_contract_groups_by_id={
            "global_macro_core": (
                "instrument",
                "observation_spec",
                "binding",
                "universe_policy",
            ),
            "core_crypto": (
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
            "core_crypto": {},
        },
    )

    decision_summaries = {
        (item.subject_set_id, item.aggregation_kind): item
        for item in result_set.decision_summaries
    }

    assert set(decision_summaries) == {
        ("global_macro_core", "active_equal_mean"),
        ("global_macro_core", "corr_weighted_mean"),
        ("core_crypto", "active_equal_mean"),
    }

    global_active = decision_summaries[("global_macro_core", "active_equal_mean")]
    assert global_active.conditions == 2
    assert global_active.wins == 1
    assert global_active.negative_conditions == 0
    assert global_active.mean_net == pytest.approx(0.07)
    assert global_active.worst_net == pytest.approx(0.04)
    assert global_active.mean_drawdown == pytest.approx(0.04)
    assert global_active.mean_gross_notional == pytest.approx(0.85)
    assert global_active.mean_net_notional == pytest.approx(0.15)
    assert global_active.mean_long_notional == pytest.approx(0.50)
    assert global_active.mean_short_notional == pytest.approx(0.35)
    assert global_active.mean_traded_notional == pytest.approx(0.45)
    assert global_active.total_cost_notional == pytest.approx(0.04)
    assert global_active.total_funding_cost_notional == pytest.approx(0.01)
    assert global_active.total_borrow_cost_notional == pytest.approx(0.008)
    assert global_active.total_roll_cost_notional == pytest.approx(0.006)
    assert global_active.subject_set_contract_groups == (
        "instrument",
        "observation_spec",
        "binding",
        "universe_policy",
    )
    assert global_active.universe_policy_fields == {
        "base_currency": "USD",
        "trading_calendar": "24x7",
        "benchmark_id": "global_macro_core",
    }

    global_corr = decision_summaries[("global_macro_core", "corr_weighted_mean")]
    assert global_corr.conditions == 2
    assert global_corr.wins == 1
    assert global_corr.negative_conditions == 0
    assert global_corr.mean_net == pytest.approx(0.06)
    assert global_corr.mean_drawdown == pytest.approx(0.03)

    crypto_active = decision_summaries[("core_crypto", "active_equal_mean")]
    assert crypto_active.conditions == 1
    assert crypto_active.wins == 1
    assert crypto_active.negative_conditions == 1
    assert crypto_active.mean_net == pytest.approx(-0.02)


def test_validation_result_set_roundtrip_preserves_structured_outputs():
    from alpha_os.validation_result_set import ValidationResultSet, build_validation_result_set

    signal_results = [
        SimpleNamespace(signal_id="momentum_1d", corr=0.20, mmc=0.05),
        SimpleNamespace(signal_id="momentum_1d", corr=-0.10, mmc=None),
    ]
    meta_results = [
        SimpleNamespace(
            date_range_label="recent",
            target_id="residual_return_3d",
            window_size=20,
            aggregation_kind="active_equal_mean",
            corr=0.20,
        ),
        SimpleNamespace(
            date_range_label="recent",
            target_id="residual_return_3d",
            window_size=20,
            aggregation_kind="corr_weighted_mean",
            corr=0.10,
        ),
    ]
    decision_results = [
        SimpleNamespace(
            date_range_label="recent",
            target_id="residual_return_3d",
            subject_set_id="global_macro_core",
            window_size=20,
            aggregation_kind="active_equal_mean",
            net_return_total=0.12,
            max_drawdown=0.04,
            mean_gross_notional_exposure=0.80,
            mean_net_notional_exposure=0.10,
            mean_long_notional_exposure=0.45,
            mean_short_notional_exposure=0.35,
            mean_traded_notional=0.30,
            cost_notional_total=0.01,
            funding_cost_notional_total=0.004,
            borrow_cost_notional_total=0.003,
            roll_cost_notional_total=0.002,
        ),
    ]

    result_set = build_validation_result_set(
        signal_results,
        meta_results,
        decision_results,
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
    restored = ValidationResultSet.from_document(result_set.to_document())

    assert restored == result_set
    assert restored.signal_summaries[0].signal_id == "momentum_1d"
    assert restored.signal_summaries[0].positive_corr == 1
    assert restored.meta_summaries[0].aggregation_kind == "active_equal_mean"
    assert restored.decision_summaries[0].subject_set_id == "global_macro_core"
    assert restored.decision_summaries[0].subject_set_contract_groups == (
        "instrument",
        "observation_spec",
        "binding",
        "universe_policy",
    )
    assert restored.decision_summaries[0].universe_policy_fields == {
        "base_currency": "USD",
        "trading_calendar": "24x7",
        "benchmark_id": "global_macro_core",
    }


def test_validation_result_set_meta_wins_use_corr_then_name_tiebreak():
    from alpha_os.validation_result_set import build_validation_result_set

    meta_results = [
        SimpleNamespace(
            date_range_label="recent",
            target_id="residual_return_3d",
            window_size=20,
            aggregation_kind="active_equal_mean",
            corr=0.20,
        ),
        SimpleNamespace(
            date_range_label="recent",
            target_id="residual_return_3d",
            window_size=20,
            aggregation_kind="corr_weighted_mean",
            corr=0.20,
        ),
        SimpleNamespace(
            date_range_label="stress",
            target_id="residual_return_3d",
            window_size=20,
            aggregation_kind="active_equal_mean",
            corr=0.10,
        ),
        SimpleNamespace(
            date_range_label="stress",
            target_id="residual_return_3d",
            window_size=20,
            aggregation_kind="corr_weighted_mean",
            corr=0.30,
        ),
    ]

    result_set = build_validation_result_set([], meta_results, [])
    meta_summaries = {item.aggregation_kind: item for item in result_set.meta_summaries}

    assert meta_summaries["active_equal_mean"].conditions == 2
    assert meta_summaries["active_equal_mean"].wins == 1
    assert meta_summaries["active_equal_mean"].mean_corr == pytest.approx(0.15)
    assert meta_summaries["corr_weighted_mean"].conditions == 2
    assert meta_summaries["corr_weighted_mean"].wins == 1
    assert meta_summaries["corr_weighted_mean"].mean_corr == pytest.approx(0.25)


def test_cross_instrument_outcome_keeps_only_scalar_metrics():
    from alpha_os.cross_instrument_outcome import build_cross_instrument_outcome

    outcome = build_cross_instrument_outcome(
        metric_group_results=(
            SimpleNamespace(
                metric_group_name="decision_quality",
                source="native_plan",
                metrics={
                    "mean_decision_net_return": 0.12,
                    "passed": True,
                    "ignored_list": [1, 2, 3],
                    "ignored_dict": {"value": 1},
                },
            ),
        ),
        failure_finding_groups=(
            SimpleNamespace(
                metric_group_name="decision_quality",
                source="native_plan",
                findings=(
                    SimpleNamespace(label="tail_a", severity=0.25),
                    SimpleNamespace(label="tail_b", severity=0.10),
                ),
            ),
        ),
    )

    assert outcome.metric_group_outcomes[0].metrics == {
        "mean_decision_net_return": 0.12,
        "passed": True,
    }
    assert outcome.failure_finding_outcomes[0].finding_count == 2
    assert outcome.failure_finding_outcomes[0].max_severity == 0.25


def test_report_metric_contract_matches_sample_outcome_keys():
    from alpha_os.cross_instrument_contract import (
        default_evaluation_report_cross_instrument_contract,
    )
    from alpha_os.cross_instrument_outcome import build_cross_instrument_outcome

    contract = default_evaluation_report_cross_instrument_contract()
    contracts = {
        (item.outcome_kind, item.metric_group_name): item.metric_fields
        for item in contract.metric_contracts
    }
    outcome = build_cross_instrument_outcome(
        metric_group_results=(
            SimpleNamespace(
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
        ),
        failure_finding_groups=(
            SimpleNamespace(
                metric_group_name="decision_quality",
                source="native_plan",
                findings=(SimpleNamespace(label="tail", severity=0.25),),
            ),
        ),
    )

    assert tuple(outcome.metric_group_outcomes[0].metrics.keys()) == contracts[
        ("metric_group_outcome", "decision_quality")
    ]
    assert contracts[("failure_finding_outcome", "decision_quality")] == (
        "finding_count",
        "max_severity",
    )
