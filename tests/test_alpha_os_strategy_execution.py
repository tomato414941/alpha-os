from __future__ import annotations


def _build_trading_strategy(
    *,
    strategy_id: str,
    label: str,
    subject_set_id: str | None = None,
    target_id: str | None = None,
    sizing_method: str | None = None,
    long_only: bool | None = None,
    gross_exposure_cap: float | None = None,
    asset_class_weight_caps: dict[str, float] | None = None,
    cluster_weight_caps: dict[str, float] | None = None,
    created_at: str = "2026-04-08T00:00:00Z",
):
    from alpha_os.trading_strategy import TradingStrategySpec
    from alpha_os.portfolio_construction_config import (
        PortfolioConstructionSizingSpec,
        PortfolioConstructionSpec,
    )

    return TradingStrategySpec(
        strategy_id=strategy_id,
        label=label,
        subject_set_id=subject_set_id,
        target_id=target_id,
        portfolio_construction=PortfolioConstructionSpec(
            sizing_policy=PortfolioConstructionSizingSpec(
                sizing_method=sizing_method or "equal_weight",
            ),
            direction_mode=(
                "long_only"
                if long_only is True
                else "long_short"
                if long_only is False
                else None
            ),
            gross_exposure_cap=gross_exposure_cap,
            asset_class_weight_caps=(
                {} if asset_class_weight_caps is None else dict(asset_class_weight_caps)
            ),
            cluster_weight_caps=(
                {} if cluster_weight_caps is None else dict(cluster_weight_caps)
            ),
        ),
        created_at=created_at,
    )


def test_trading_strategy_exposes_policy_hierarchy():
    trading_strategy = _build_trading_strategy(
        strategy_id="strategy:test",
        label="Strategy Test",
        subject_set_id="core_crypto",
        target_id="residual_return_3d",
        sizing_method="equal_weight",
        long_only=True,
        gross_exposure_cap=1.5,
        asset_class_weight_caps={"equity_index": 0.6},
        cluster_weight_caps={"eq_us": 0.25},
    )

    assert trading_strategy.strategy_id == "strategy:test"
    assert trading_strategy.subject_set_id == "core_crypto"
    assert trading_strategy.target_id == "residual_return_3d"
    assert trading_strategy.portfolio_construction.sizing_method == "equal_weight"
    assert trading_strategy.portfolio_construction.long_only is True
    assert trading_strategy.portfolio_construction.gross_exposure_cap == 1.5
    assert trading_strategy.portfolio_construction.asset_class_weight_caps == {
        "equity_index": 0.6
    }
    assert trading_strategy.portfolio_construction.cluster_weight_caps == {
        "eq_us": 0.25
    }
def test_trading_strategy_ignores_removed_top_k_document_field():
    from alpha_os.trading_strategy import TradingStrategySpec

    strategy = TradingStrategySpec.from_document(
        {
            "strategy_id": "strategy:test",
            "label": "Test",
            "subject_set_id": "core_crypto",
            "target_id": "residual_return_3d",
            "portfolio_construction": {},
            "selection_kind": "top_k",
            "top_k": 4,
            "created_at": "2026-04-08T00:00:00Z",
        }
    )

    assert "top_k" not in strategy.to_document()


def test_trading_strategy_spec_round_trips_through_document():
    from alpha_os.trading_strategy import TradingStrategySpec

    strategy = _build_trading_strategy(
        strategy_id="strategy:test",
        label="Strategy Test",
        subject_set_id="core_crypto",
        target_id="residual_return_3d",
        sizing_method="equal_weight",
        long_only=True,
        gross_exposure_cap=1.5,
        asset_class_weight_caps={"equity_index": 0.6},
        cluster_weight_caps={"eq_us": 0.25},
    )

    round_tripped = TradingStrategySpec.from_document(strategy.to_document())

    assert round_tripped.strategy_id == strategy.strategy_id
    assert round_tripped.label == strategy.label
    assert round_tripped.created_at == strategy.created_at
    assert "portfolio" not in strategy.to_document()
    assert "portfolio_policy" not in strategy.to_document()
    assert round_tripped.portfolio_construction == strategy.portfolio_construction


def test_trading_strategy_contract_accepts_black_box_decision_component():
    from alpha_os.trading_strategy import (
        TradingStrategy,
        TradingStrategyInput,
        TradingStrategyOutput,
    )

    class FixedWeightStrategy:
        def decide(self, strategy_input: TradingStrategyInput) -> TradingStrategyOutput:
            return TradingStrategyOutput()

    strategy: TradingStrategy = FixedWeightStrategy()
    decision = strategy.decide(TradingStrategyInput())

    assert isinstance(decision, TradingStrategyOutput)
