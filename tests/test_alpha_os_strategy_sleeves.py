from __future__ import annotations

import pytest


def _sleeve_composition():
    from alpha_os.strategy_sleeves import (
        StrategySleeveCompositionSpec,
        StrategySleeveSpec,
        StrategySleeveSubjectFilterSpec,
    )

    return StrategySleeveCompositionSpec(
        sleeves=(
            StrategySleeveSpec(
                sleeve_id="trend_core",
                sleeve_kind="trend",
                signal_source_kind="trend",
                risk_budget=0.75,
            ),
            StrategySleeveSpec(
                sleeve_id="carry_satellite",
                sleeve_kind="carry",
                signal_source_kind="carry",
                risk_budget=0.25,
                subject_filter=StrategySleeveSubjectFilterSpec(
                    asset_classes=("rates",),
                ),
            ),
        )
    )


def test_strategy_sleeve_composition_round_trips_on_strategy_and_case_config():
    from alpha_os.portfolio_construction_config import (
        PortfolioConstructionSizingSpec,
        PortfolioConstructionSpec,
    )
    from alpha_os.trading_strategy import (
        ExecutionPolicySpec,
        RebalanceFrictionPolicySpec,
        StrategyPortfolioSpec,
        TradingStrategyScopeSpec,
        TradingStrategySpec,
        HoldingCostPolicySpec,
    )

    composition = _sleeve_composition()
    strategy = TradingStrategySpec(
        strategy_id="strategy:sleeve",
        label="sleeve strategy",
        scope=TradingStrategyScopeSpec(
            subject_set_id="macro",
            target_id="residual_return_1d",
        ),
        signal_discovery_id=None,
        position_rule_id="constant_hold",
        family_mix=None,
        portfolio=StrategyPortfolioSpec(
            portfolio_construction=PortfolioConstructionSpec(
                sizing_policy=PortfolioConstructionSizingSpec(
                    sizing_method="signal_weighted",
                ),
                direction_mode="long_short",
                gross_exposure_cap=1.0,
                sleeve_composition=composition,
            ),
            rebalance_friction_policy=RebalanceFrictionPolicySpec(
                turnover_friction=0.0,
                no_trade_band=0.0,
            ),
            execution_policy=ExecutionPolicySpec(market_impact_bps=0.0),
            holding_cost_policy=HoldingCostPolicySpec(),
            selection_kind="all_assets",
            top_k=None,
        ),
        created_at="2026-04-19T00:00:00Z",
    )

    restored_strategy = TradingStrategySpec.from_document(strategy.to_document())
    assert restored_strategy.sleeve_composition == composition

    construction = PortfolioConstructionSpec(sleeve_composition=composition)
    restored_construction = PortfolioConstructionSpec.from_document(
        construction.to_document()
    )
    assert restored_construction.sleeve_composition == composition


def test_strategy_sleeve_composition_blends_signals_before_sizing():
    from alpha_os.portfolio_decision import (
        ObservedPortfolioInputs,
        PortfolioDecisionInput,
        PortfolioState,
        PredictiveSignalInput,
    )
    from alpha_os.portfolio_sizing_policy import (
        SignalWeightedSizingPolicy,
        apply_portfolio_sizing_policy,
    )

    decision_input = PortfolioDecisionInput(
        portfolio_state=PortfolioState(capital_base=1.0),
        observed_inputs=ObservedPortfolioInputs(
            predictive_signals=(
                PredictiveSignalInput(
                    source_id="trend",
                    source_kind="trend",
                    subject_id="ES_future",
                    target_id="residual_return_1d",
                    value=1.0,
                ),
                PredictiveSignalInput(
                    source_id="trend",
                    source_kind="trend",
                    subject_id="ZN_future",
                    target_id="residual_return_1d",
                    value=-1.0,
                ),
                PredictiveSignalInput(
                    source_id="carry",
                    source_kind="carry",
                    subject_id="ZN_future",
                    target_id="residual_return_1d",
                    value=2.0,
                ),
            )
        ),
        sleeve_composition=_sleeve_composition(),
        subject_metadata_by_subject={
            "ES_future": {"asset_class": "equity_index"},
            "ZN_future": {"asset_class": "rates"},
        },
    )

    output = apply_portfolio_sizing_policy(
        decision_input,
        sizing_policy=SignalWeightedSizingPolicy(max_abs_weight=10.0),
    )

    weights = {target.subject_id: target.target_weight for target in output.targets}
    assert weights["ES_future"] == pytest.approx(0.75)
    assert weights["ZN_future"] == pytest.approx(-0.25)


def test_strategy_sleeve_composition_rejects_duplicate_ids_and_empty_filters():
    from alpha_os.portfolio_decision import (
        ObservedPortfolioInputs,
        PortfolioDecisionInput,
        PortfolioState,
        PredictiveSignalInput,
    )
    from alpha_os.portfolio_sizing_policy import apply_portfolio_sizing_policy
    from alpha_os.strategy_sleeves import (
        StrategySleeveCompositionSpec,
        StrategySleeveSpec,
        StrategySleeveSubjectFilterSpec,
    )

    with pytest.raises(ValueError, match="unique"):
        StrategySleeveCompositionSpec(
            sleeves=(
                StrategySleeveSpec(
                    sleeve_id="trend",
                    sleeve_kind="trend",
                    risk_budget=1.0,
                ),
                StrategySleeveSpec(
                    sleeve_id="trend",
                    sleeve_kind="trend",
                    risk_budget=1.0,
                ),
            )
        )

    composition = StrategySleeveCompositionSpec(
        sleeves=(
            StrategySleeveSpec(
                sleeve_id="rates_trend",
                sleeve_kind="trend",
                risk_budget=1.0,
                subject_filter=StrategySleeveSubjectFilterSpec(
                    asset_classes=("rates",),
                ),
            ),
        )
    )
    decision_input = PortfolioDecisionInput(
        portfolio_state=PortfolioState(capital_base=1.0),
        observed_inputs=ObservedPortfolioInputs(
            predictive_signals=(
                PredictiveSignalInput(
                    source_id="trend",
                    source_kind="trend",
                    subject_id="ES_future",
                    target_id="residual_return_1d",
                    value=1.0,
                ),
            )
        ),
        sleeve_composition=composition,
        subject_metadata_by_subject={
            "ES_future": {"asset_class": "equity_index"},
        },
    )

    with pytest.raises(ValueError, match="no eligible subjects"):
        apply_portfolio_sizing_policy(decision_input)


def test_evaluation_task_result_serializes_sleeve_attribution():
    from alpha_os.evaluation_report import EvaluationTaskResult
    from alpha_os.strategy_sleeves import SleeveAttributionSummary

    summary = EvaluationTaskResult(
        evaluation_task_id="case:sleeve",
        strategy_id="strategy:sleeve",
        sleeve_attribution_summaries=(
            SleeveAttributionSummary(
                sleeve_id="trend_core",
                sleeve_kind="trend",
                risk_budget=1.0,
                subject_count=2,
                mean_signal=0.25,
                mean_abs_signal=0.75,
                mean_gross_notional_exposure=1.2,
            ),
        ),
    )

    restored = EvaluationTaskResult.from_document(summary.to_document())
    assert restored.sleeve_attribution_summaries == summary.sleeve_attribution_summaries
