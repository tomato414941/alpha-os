from __future__ import annotations


def _build_trading_strategy(
    *,
    strategy_id: str,
    label: str,
    subject_set_id: str | None = None,
    target_id: str | None = None,
    signal_discovery_id: str | None = None,
    position_rule_id: str = "constant_hold",
    family_mix: str | None = None,
    selection_kind: str = "all_assets",
    sizing_method: str | None = None,
    rebalance: str | None = None,
    long_only: bool | None = None,
    top_k: int | None = None,
    gross_exposure_cap: float | None = None,
    asset_class_weight_caps: dict[str, float] | None = None,
    cluster_weight_caps: dict[str, float] | None = None,
    market_impact_bps: float | None = None,
    fee_bps: float | None = None,
    bid_ask_spread_bps: float | None = None,
    funding_bps_per_step: float | None = None,
    borrow_fee_bps_per_step: float | None = None,
    turnover_cost_rate: float | None = None,
    created_at: str = "2026-04-08T00:00:00Z",
):
    from alpha_os.evaluation_cost_config import TradingEnvironment
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
        signal_discovery_id=signal_discovery_id,
        position_rule_id=position_rule_id,
        family_mix=family_mix,
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
        trading_environment=TradingEnvironment(
            turnover_cost_rate=(
                0.0 if turnover_cost_rate is None else turnover_cost_rate
            ),
            market_impact_bps=0.0 if market_impact_bps is None else market_impact_bps,
            fee_bps=0.0 if fee_bps is None else fee_bps,
            bid_ask_spread_bps=(
                0.0 if bid_ask_spread_bps is None else bid_ask_spread_bps
            ),
            funding_bps_per_step=(
                0.0 if funding_bps_per_step is None else funding_bps_per_step
            ),
            borrow_fee_bps_per_step=(
                0.0 if borrow_fee_bps_per_step is None else borrow_fee_bps_per_step
            ),
        ),
        created_at=created_at,
        rebalance_interval_steps=(
            int(rebalance[len("every_") : -len("_steps")])
            if isinstance(rebalance, str)
            and rebalance.startswith("every_")
            and rebalance.endswith("_steps")
            else 1
        ),
        selection_kind=selection_kind,
        top_k=top_k,
    )


def test_trading_strategy_exposes_policy_hierarchy():
    trading_strategy = _build_trading_strategy(
        strategy_id="strategy:test",
        label="Strategy Test",
        subject_set_id="core_crypto",
        target_id="residual_return_3d",
        signal_discovery_id="discovery:core",
        family_mix="relative_strength",
        sizing_method="equal_weight",
        rebalance="every_5_steps",
        long_only=True,
        top_k=5,
        gross_exposure_cap=1.5,
        asset_class_weight_caps={"equity_index": 0.6},
        cluster_weight_caps={"eq_us": 0.25},
        market_impact_bps=5.0,
        fee_bps=2.0,
        bid_ask_spread_bps=3.0,
        funding_bps_per_step=1.5,
        borrow_fee_bps_per_step=2.5,
        turnover_cost_rate=0.1,
    )

    assert trading_strategy.strategy_id == "strategy:test"
    assert trading_strategy.subject_set_id == "core_crypto"
    assert trading_strategy.target_id == "residual_return_3d"
    assert (
        trading_strategy.signal_discovery_id
        == "discovery:core"
    )
    assert (
        trading_strategy.family_mix
        == "relative_strength"
    )
    assert trading_strategy.selection_kind == "all_assets"
    assert trading_strategy.top_k == 5
    assert trading_strategy.portfolio_construction.sizing_method == "equal_weight"
    assert trading_strategy.rebalance_interval_steps == 5
    assert trading_strategy.portfolio_construction.long_only is True
    assert trading_strategy.portfolio_construction.gross_exposure_cap == 1.5
    assert trading_strategy.portfolio_construction.asset_class_weight_caps == {
        "equity_index": 0.6
    }
    assert trading_strategy.portfolio_construction.cluster_weight_caps == {
        "eq_us": 0.25
    }
    assert trading_strategy.trading_environment.turnover_cost_rate == 0.1
    assert trading_strategy.trading_environment.market_impact_bps == 5.0
    assert trading_strategy.trading_environment.fee_bps == 2.0
    assert trading_strategy.trading_environment.bid_ask_spread_bps == 3.0
    assert trading_strategy.trading_environment.funding_bps_per_step == 1.5
    assert trading_strategy.trading_environment.borrow_fee_bps_per_step == 2.5

def test_trading_strategy_top_k_is_serialized_with_selection_policy():
    trading_strategy = _build_trading_strategy(
        strategy_id="strategy:test",
        label="Test",
        selection_kind="top_k",
        top_k=3,
    )

    document = trading_strategy.to_document()

    assert document["selection_kind"] == "top_k"
    assert document["top_k"] == 3


def test_trading_strategy_top_k_round_trips_from_document():
    from alpha_os.trading_strategy import TradingStrategySpec

    strategy = TradingStrategySpec.from_document(
        {
            "strategy_id": "strategy:test",
            "label": "Test",
            "subject_set_id": "core_crypto",
            "target_id": "residual_return_3d",
            "signal_discovery_id": None,
            "position_rule_id": "constant_hold",
            "family_mix": None,
            "portfolio_construction": {},
            "trading_environment": {},
            "rebalance_interval_steps": 1,
            "selection_kind": "top_k",
            "top_k": 4,
            "created_at": "2026-04-08T00:00:00Z",
        }
    )

    assert strategy.top_k == 4
    assert strategy.selection_kind == "top_k"


def test_trading_strategy_spec_round_trips_through_document():
    from alpha_os.trading_strategy import TradingStrategySpec

    strategy = _build_trading_strategy(
        strategy_id="strategy:test",
        label="Strategy Test",
        subject_set_id="core_crypto",
        target_id="residual_return_3d",
        signal_discovery_id="discovery:core",
        family_mix="relative_strength",
        sizing_method="equal_weight",
        rebalance="every_5_steps",
        long_only=True,
        top_k=5,
        gross_exposure_cap=1.5,
        asset_class_weight_caps={"equity_index": 0.6},
        cluster_weight_caps={"eq_us": 0.25},
        market_impact_bps=5.0,
        fee_bps=2.0,
        bid_ask_spread_bps=3.0,
        funding_bps_per_step=1.5,
        borrow_fee_bps_per_step=2.5,
        turnover_cost_rate=0.1,
    )

    round_tripped = TradingStrategySpec.from_document(strategy.to_document())

    assert round_tripped.strategy_id == strategy.strategy_id
    assert round_tripped.label == strategy.label
    assert round_tripped.created_at == strategy.created_at
    assert "portfolio" not in strategy.to_document()
    assert "portfolio_policy" not in strategy.to_document()
    assert round_tripped.portfolio_construction == strategy.portfolio_construction
    assert round_tripped.trading_environment == strategy.trading_environment


def test_trading_strategy_contract_accepts_black_box_decision_component():
    from alpha_os.portfolio_decision import (
        PortfolioDecisionInput,
        PortfolioDecisionOutput,
        PortfolioTarget,
    )
    from alpha_os.trading_strategy_contract import TradingStrategy

    class FixedWeightStrategy:
        def decide(self, decision_input: PortfolioDecisionInput) -> PortfolioDecisionOutput:
            return PortfolioDecisionOutput(
                portfolio_id=decision_input.portfolio_id,
                as_of=decision_input.as_of,
                targets=(
                    PortfolioTarget(
                        subject_id="BTC",
                        target_weight=1.0,
                        position_delta=1.0,
                    ),
                ),
            )

    strategy: TradingStrategy = FixedWeightStrategy()
    decision = strategy.decide(
        PortfolioDecisionInput(portfolio_id="portfolio:test", as_of="2026-04-08")
    )

    assert decision.portfolio_id == "portfolio:test"
    assert decision.targets[0].subject_id == "BTC"
    assert decision.targets[0].target_weight == 1.0
