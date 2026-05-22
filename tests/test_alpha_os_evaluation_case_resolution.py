from __future__ import annotations

from types import SimpleNamespace

from alpha_os.strategy_variant import (
    StrategyVariantConfig,
    derive_trading_strategy_from_signal_discovery,
    overridden_strategy_variant_config,
)
from alpha_os.evaluation_cost_config import (
    EvaluationRebalanceFrictionPolicySpec,
    MarketAssumptions,
)
from alpha_os.evaluation_spec import (
    EvaluationDateRange,
    EvaluationFold,
    EvaluationSpec,
)
from alpha_os.portfolio_construction_config import (
    PortfolioConstructionSizingSpec,
    PortfolioConstructionSpec,
)
from alpha_os.signal_discovery import SignalDiscoverySpec


def _make_evaluation_trading_config(
    *,
    sizing_method: str = "signal_weighted",
    sizing_engine: str | None = None,
    rebalance_interval_steps: int = 1,
    long_only: bool = False,
    direction_mode: str | None = None,
    top_k: int | None = None,
    gross_exposure_cap: float | None = None,
    target_vol: float | None = None,
    gross_leverage_cap: float | None = None,
    net_exposure_target: float | None = None,
    asset_class_weight_caps: dict[str, float] | None = None,
    cluster_weight_caps: dict[str, float] | None = None,
) -> StrategyVariantConfig:
    return StrategyVariantConfig(
        portfolio_construction=PortfolioConstructionSpec(
            sizing_policy=PortfolioConstructionSizingSpec(
                sizing_method=sizing_method,
                sizing_engine=sizing_engine,
            ),
            rebalance_interval_steps=rebalance_interval_steps,
            long_only=long_only,
            direction_mode=direction_mode,
            gross_exposure_cap=gross_exposure_cap,
            target_vol=target_vol,
            gross_leverage_cap=gross_leverage_cap,
            net_exposure_target=net_exposure_target,
            asset_class_weight_caps={} if asset_class_weight_caps is None else asset_class_weight_caps,
            cluster_weight_caps={} if cluster_weight_caps is None else cluster_weight_caps,
        ),
        rebalance_friction_policy=EvaluationRebalanceFrictionPolicySpec(),
        market_assumptions=MarketAssumptions(),
        top_k=top_k,
    )


def _make_signal_discovery(
    *,
    signal_discovery_id: str = "global_macro_search",
    subject_set_id: str = "global_macro_core",
    target_id: str = "residual_return_5d",
    signal_spec_id: str = "momentum_1d",
):
    definition = SignalDiscoverySpec(
        signal_discovery_id=signal_discovery_id,
        subject_set_id=subject_set_id,
        signal_spec_ids=(signal_spec_id,),
        target_id=target_id,
    )
    return SimpleNamespace(
        signal_discovery_id=signal_discovery_id,
        definition=definition,
    )


def _make_evaluation_spec_with_two_folds() -> EvaluationSpec:
    return EvaluationSpec(
        execution_range=EvaluationDateRange(
            label="full",
            start_date="2026-01-01",
            end_date="2026-04-30",
        ),
        evaluation_folds=(
            EvaluationFold(
                label="fold_1",
                execution_range=EvaluationDateRange(
                    label="fold_1_execution",
                    start_date="2026-01-01",
                    end_date="2026-02-28",
                ),
                evaluation_date_ranges=(
                    EvaluationDateRange(
                        label="fold_1_eval",
                        start_date="2026-03-01",
                        end_date="2026-03-31",
                    ),
                ),
            ),
            EvaluationFold(
                label="fold_2",
                execution_range=EvaluationDateRange(
                    label="fold_2_execution",
                    start_date="2026-02-01",
                    end_date="2026-03-31",
                ),
                evaluation_date_ranges=(
                    EvaluationDateRange(
                        label="fold_2_eval",
                        start_date="2026-04-01",
                        end_date="2026-04-30",
                    ),
                ),
            ),
        ),
        metric_group_names=("decision_quality",),
        metric_windows=(20,),
    )


def test_overridden_strategy_variant_config_returns_same_config_without_override():
    config = _make_evaluation_trading_config()

    resolved = overridden_strategy_variant_config(
        config,
        sizing_method=None,
        sizing_engine=None,
    )

    assert resolved is config


def test_overridden_strategy_variant_config_preserves_risk_contract():
    config = _make_evaluation_trading_config(
        sizing_method="signal_weighted",
        sizing_engine="rule_based",
        long_only=False,
        top_k=5,
        gross_exposure_cap=1.5,
        target_vol=0.18,
        gross_leverage_cap=1.8,
        net_exposure_target=0.0,
        asset_class_weight_caps={"commodity": 0.4},
        cluster_weight_caps={"rates": 0.35},
    )

    resolved = overridden_strategy_variant_config(
        config,
        sizing_method="hierarchical_risk_parity",
        sizing_engine=None,
    )

    construction = resolved.portfolio_construction
    assert construction.sizing_method == "hierarchical_risk_parity"
    assert construction.sizing_engine == "history_based"
    assert construction.long_only is False
    assert resolved.top_k == 5
    assert construction.gross_exposure_cap == 1.5
    assert construction.target_vol == 0.18
    assert construction.gross_leverage_cap == 1.8
    assert construction.net_exposure_target == 0.0
    assert construction.asset_class_weight_caps == {"commodity": 0.4}
    assert construction.cluster_weight_caps == {"rates": 0.35}


def test_overridden_strategy_variant_config_can_override_direction_mode():
    config = _make_evaluation_trading_config(long_only=False)

    resolved = overridden_strategy_variant_config(
        config,
        sizing_method=None,
        sizing_engine=None,
        direction_mode="short_only",
    )

    assert resolved.portfolio_construction.direction_mode == "short_only"
    assert resolved.portfolio_construction.long_only is False


def test_overridden_strategy_variant_config_can_override_sizing_engine_only():
    config = _make_evaluation_trading_config(
        sizing_method="signal_weighted",
        sizing_engine="rule_based",
    )

    resolved = overridden_strategy_variant_config(
        config,
        sizing_method=None,
        sizing_engine="optimizer",
    )

    assert resolved.portfolio_construction.sizing_method == "signal_weighted"
    assert resolved.portfolio_construction.sizing_engine == "optimizer"


def test_derived_trading_strategy_uses_top_k_selection_when_top_k_is_set():
    config = _make_evaluation_trading_config(
        sizing_method="equal_weight",
        sizing_engine="history_based",
        rebalance_interval_steps=5,
        long_only=True,
        top_k=3,
        gross_exposure_cap=0.8,
    )

    strategy = derive_trading_strategy_from_signal_discovery(
        signal_discovery=_make_signal_discovery(target_id="residual_return_3d"),
        variant_config=config,
        created_at="2026-04-17T00:00:00Z",
    )

    assert strategy.selection_kind == "top_k"
    assert strategy.portfolio.top_k == 3


def test_derived_trading_strategy_preserves_risk_policy_constraints():
    config = _make_evaluation_trading_config(
        sizing_method="signal_weighted",
        sizing_engine="rule_based",
        long_only=False,
        gross_exposure_cap=1.2,
        target_vol=0.18,
        gross_leverage_cap=1.5,
        net_exposure_target=0.0,
        asset_class_weight_caps={"commodity": 0.35},
        cluster_weight_caps={"rates_us": 0.3},
    )

    strategy = derive_trading_strategy_from_signal_discovery(
        signal_discovery=_make_signal_discovery(target_id="residual_return_3d"),
        variant_config=config,
        created_at="2026-04-17T00:00:00Z",
    )

    construction = strategy.portfolio.portfolio_construction
    assert construction.long_only is False
    assert construction.gross_exposure_cap == 1.2
    assert construction.target_vol == 0.18
    assert construction.gross_leverage_cap == 1.5
    assert construction.net_exposure_target == 0.0
    assert construction.asset_class_weight_caps == {"commodity": 0.35}
    assert construction.cluster_weight_caps == {"rates_us": 0.3}
