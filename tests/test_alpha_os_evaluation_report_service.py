from __future__ import annotations


def test_evaluation_task_contract_fields_use_portfolio_construction_risk_policy():
    from alpha_os.evaluation_cost_config import (
        EvaluationRebalanceFrictionPolicySpec,
        ExecutionCostAssumptionsSpec,
        HoldingCostAssumptionsSpec,
    )
    from alpha_os.portfolio_construction_config import (
        PortfolioConstructionSizingSpec,
        PortfolioConstructionSpec,
        PortfolioRiskBudgetSpec,
    )
    from alpha_os.evaluation_report_service import (
        build_report_evaluation_task_contract_fields,
    )
    from alpha_os.portfolio_decision import SubjectSet, UniversePolicySpec
    from alpha_os.strategy_sleeves import StrategySleeveCompositionSpec

    portfolio_construction = PortfolioConstructionSpec(
        sizing_policy=PortfolioConstructionSizingSpec(
            sizing_method="hierarchical_risk_parity",
            sizing_engine="history_based",
        ),
        rebalance_interval_steps=1,
        long_only=False,
        gross_exposure_cap=1.5,
        target_vol=0.18,
        gross_leverage_cap=1.5,
        net_exposure_target=0.0,
        risk_budget=PortfolioRiskBudgetSpec(
            risk_normalization_mode="gross",
            target_gross_exposure=0.5,
            allow_releverage=True,
        ),
        asset_class_weight_caps={"rates": 0.55},
        cluster_weight_caps={"rates_us": 0.30},
        sleeve_composition=StrategySleeveCompositionSpec.from_document(
            {
                "sleeves": [
                    {
                        "sleeve_id": "trend_core",
                        "sleeve_kind": "trend",
                        "risk_budget": 1.0,
                    }
                ]
            }
        ),
    )
    subject_set = SubjectSet(
        subject_set_id="global_macro_tradeable_daily_10y",
        universe_policy=UniversePolicySpec(
            base_currency="USD",
            trading_calendar="multi_venue",
            benchmark_id="global_macro_tradeable_daily_10y",
        ),
    )

    fields = build_report_evaluation_task_contract_fields(
        portfolio_construction,
        rebalance_friction_policy=EvaluationRebalanceFrictionPolicySpec(
            turnover_friction=0.001,
            no_trade_band=0.0,
        ),
        execution_cost_assumptions=ExecutionCostAssumptionsSpec(
            market_impact_bps=4.0,
            fee_bps=0.5,
        ),
        holding_cost_assumptions=HoldingCostAssumptionsSpec(
            funding_bps_per_step=0.5,
            borrow_fee_bps_per_step=0.25,
        ),
        subject_set=subject_set,
        subject_set_id="global_macro_tradeable_daily_10y",
    )

    assert fields["sizing"] == "hierarchical_risk_parity"
    assert fields["target_vol"] == 0.18
    assert fields["gross_leverage_cap"] == 1.5
    assert fields["net_exposure_target"] == 0.0
    assert fields["risk_normalization_mode"] == "gross"
    assert fields["target_gross_exposure"] == 0.5
    assert fields["allow_releverage"] == "true"
    assert fields["sleeve_count"] == 1
    assert fields["sleeves"] == "trend_core:trend:1.0"
    assert fields["base_currency"] == "USD"
    assert fields["trading_calendar"] == "multi_venue"
    assert "sizing_time:target_vol" in str(fields["constraint_stages"])
    assert "gross_leverage_cap,net_exposure_target" in str(
        fields["constraint_stages"]
    )


def test_evaluation_task_contract_fields_use_strategy_portfolio_selection():
    from alpha_os.evaluation_cost_config import (
        EvaluationRebalanceFrictionPolicySpec,
        ExecutionCostAssumptionsSpec,
        HoldingCostAssumptionsSpec,
    )
    from alpha_os.portfolio_construction_config import PortfolioConstructionSpec
    from alpha_os.evaluation_report_service import (
        build_report_evaluation_task_contract_fields,
    )

    fields = build_report_evaluation_task_contract_fields(
        PortfolioConstructionSpec(),
        rebalance_friction_policy=EvaluationRebalanceFrictionPolicySpec(),
        execution_cost_assumptions=ExecutionCostAssumptionsSpec(),
        holding_cost_assumptions=HoldingCostAssumptionsSpec(),
        selection_kind="top_k",
        top_k=3,
    )

    assert fields["selection"] == "top_k"
    assert fields["top_k"] == 3


def test_hold_baseline_contract_fields_suppress_active_portfolio_noise():
    from alpha_os.evaluation_cost_config import (
        EvaluationRebalanceFrictionPolicySpec,
        ExecutionCostAssumptionsSpec,
        HoldingCostAssumptionsSpec,
    )
    from alpha_os.portfolio_construction_config import (
        PortfolioConstructionSizingSpec,
        PortfolioConstructionSpec,
    )
    from alpha_os.evaluation_report_service import (
        build_report_evaluation_task_contract_fields,
    )

    fields = build_report_evaluation_task_contract_fields(
        PortfolioConstructionSpec(
            construction_kind="hold_baseline",
            sizing_policy=PortfolioConstructionSizingSpec(
                sizing_method="equal_weight",
                sizing_engine="history_based",
            ),
            rebalance_interval_steps=252,
            long_only=True,
            active_overlay=None,
            gross_exposure_cap=1.0,
            gross_leverage_cap=1.0,
            net_exposure_target=1.0,
        ),
        rebalance_friction_policy=EvaluationRebalanceFrictionPolicySpec(
            turnover_friction=0.001,
            no_trade_band=0.0,
        ),
        execution_cost_assumptions=ExecutionCostAssumptionsSpec(
            market_impact_bps=4.0,
            fee_bps=0.5,
        ),
        holding_cost_assumptions=HoldingCostAssumptionsSpec(
            funding_bps_per_step=0.5,
            borrow_fee_bps_per_step=0.25,
        ),
        subject_set_id="global_macro_tradeable_daily_10y",
    )

    assert fields["construction_kind"] == "hold_baseline"
    assert fields["holding_style"] == "equal_weight_hold"
    assert "active_overlay" not in fields
    assert "sizing_family" not in fields
    assert "target_vol" not in fields
    assert "target_gross_exposure" not in fields


def test_hold_baseline_portfolio_construction_rejects_active_overlay():
    import pytest

    from alpha_os.portfolio_construction_config import (
        PortfolioConstructionSizingSpec,
        PortfolioConstructionSpec,
    )

    with pytest.raises(ValueError, match="must not define active_overlay"):
        PortfolioConstructionSpec(
            construction_kind="hold_baseline",
            sizing_policy=PortfolioConstructionSizingSpec(
                sizing_method="equal_weight",
                sizing_engine="history_based",
            ),
            long_only=True,
        )
