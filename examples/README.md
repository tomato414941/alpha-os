# Examples

These examples are concrete sketches, not alpha-os public API.

The maintained package currently exposes only the `TradingStrategy` contract:

```text
decide(input) -> output
```

The example-local names such as `MarketObservation`, `PortfolioTarget`,
`TradingIntent`, `Order`, and `MarketBacktestWorld` are intentionally not
defined in the package. They show possible shapes a concrete strategy may
choose.

Read them in this order:

1. `trading_strategy_backtest.py`
   - a strategy interacts with a market world through
     `strategy.decide(...)` and `world.step(action)`
   - input/output shape: `MarketObservation -> PortfolioTarget`
2. `trading_strategy_execution_intent.py`
   - a strategy output can include execution intent, not only portfolio targets
   - input/output shape: `RiskObservation -> TradingIntent`

Other concrete TradingStrategy input/output shapes:

- `trading_strategy_orders.py`
  - input/output shape: `MarketObservation -> list[Order]`
- `trading_strategy_hold_rebalance.py`
  - input/output shape: `PortfolioObservation -> Hold | Rebalance`
- `trading_strategy_hedge_action.py`
  - input/output shape: `PortfolioRiskObservation -> HedgeAction | None`
- `trading_strategy_spread_trade.py`
  - input/output shape: `SpreadObservation -> SpreadTradeIntent | None`
- `trading_strategy_stateful.py`
  - input/output shape: `PriceObservation -> PositionAction`
  - the strategy keeps internal state between calls
- `trading_strategy_execution_orders.py`
  - input/output shape: `BrokerObservation -> list[ExecutionOrder]`

Strategy internals can have their own shapes. These are not TradingStrategy
examples:

- `alpha_model_score.py`
  - shape: `FeatureBatch -> AlphaScore`
- `risk_model_exposure.py`
  - shape: `PortfolioSnapshot -> RiskEstimate`
- `portfolio_allocator_from_scores.py`
  - shape: `AlphaScore -> PortfolioTarget`
- `execution_order_slicer.py`
  - shape: `ParentOrder -> tuple[ChildOrder, ...]`

In ML/RL terms, `TradingStrategy` is policy-like. The concrete observation,
action, world, and backtest shapes remain strategy- or example-specific until
multiple real use cases justify a package-level abstraction.
