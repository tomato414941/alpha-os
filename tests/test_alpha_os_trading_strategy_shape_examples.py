from __future__ import annotations

import pytest

from conftest import load_example_module


def test_trading_strategy_can_return_orders():
    example = load_example_module("examples/trading_strategy_orders.py")

    orders = example.decide_orders(
        example.BuyDipOrderStrategy(
            symbol="BTC",
            reference_price=100.0,
            quantity=0.1,
        ),
        example.MarketObservation(prices={"BTC": 95.0}, cash=1_000.0),
    )

    assert orders == [example.Order(symbol="BTC", side="buy", quantity=0.1)]


def test_trading_strategy_can_return_hold_or_rebalance():
    example = load_example_module("examples/trading_strategy_hold_rebalance.py")

    decision = example.decide_rebalance(
        example.DriftAwareRebalanceStrategy(rebalance_threshold=0.1),
        example.PortfolioObservation(
            target_weights={"BTC": 0.5, "ETH": 0.5},
            current_weights={"BTC": 0.7, "ETH": 0.3},
        ),
    )

    assert decision == example.Rebalance(target_weights={"BTC": 0.5, "ETH": 0.5})


def test_trading_strategy_can_return_hedge_action():
    example = load_example_module("examples/trading_strategy_hedge_action.py")

    hedge = example.decide_hedge(
        example.ExposureHedgeStrategy(max_exposure=1.0),
        example.PortfolioRiskObservation(
            net_exposure=1.4,
            hedge_symbol="BTC-PERP",
        ),
    )

    assert hedge is not None
    assert hedge.symbol == "BTC-PERP"
    assert hedge.target_notional == pytest.approx(-0.4)


def test_trading_strategy_can_return_spread_trade_intent():
    example = load_example_module("examples/trading_strategy_spread_trade.py")

    intent = example.decide_spread_trade(
        example.MeanReversionSpreadStrategy(entry_zscore=2.0, quantity=1.0),
        example.SpreadObservation(
            long_symbol="BTC",
            short_symbol="ETH",
            spread_zscore=2.2,
        ),
    )

    assert intent == example.SpreadTradeIntent(
        legs=(
            example.SpreadLeg(symbol="BTC", side="buy", quantity=1.0),
            example.SpreadLeg(symbol="ETH", side="sell", quantity=1.0),
        ),
        reason="mean_reversion_entry",
    )


def test_trading_strategy_component_can_return_alpha_score():
    example = load_example_module("examples/trading_strategy_alpha_score_component.py")

    score = example.score_alpha(
        example.MomentumAlphaModel(),
        example.FeatureBatch(
            features_by_symbol={
                "BTC": {"return_7d": 0.04},
                "ETH": {"return_7d": -0.02},
            }
        ),
    )

    assert score == example.AlphaScore(scores={"BTC": 0.04, "ETH": -0.02})


def test_trading_strategy_can_keep_internal_state():
    example = load_example_module("examples/trading_strategy_stateful.py")

    actions = example.decide_positions(
        example.BreakoutStatefulStrategy(breakout_fraction=0.01),
        [
            example.PriceObservation(price=100.0),
            example.PriceObservation(price=102.0),
        ],
    )

    assert actions == [
        example.PositionAction(target_position=0.0),
        example.PositionAction(target_position=1.0),
    ]


def test_trading_strategy_can_return_execution_orders():
    example = load_example_module("examples/trading_strategy_execution_orders.py")

    orders = example.decide_execution_orders(
        example.FillRemainingExecutionStrategy(order_style="limit"),
        example.BrokerObservation(
            symbol="BTC",
            target_quantity=1.0,
            filled_quantity=0.25,
            last_price=100.0,
        ),
    )

    assert orders == [
        example.ExecutionOrder(
            symbol="BTC",
            side="buy",
            quantity=0.75,
            order_style="limit",
        )
    ]
