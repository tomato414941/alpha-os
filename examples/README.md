# Examples

These examples are concrete sketches, not alpha-os public API.

The maintained package currently exposes only the `TradingStrategy` contract:

```text
decide(input) -> output
```

The example-local names such as `MarketObservation`, `PortfolioAction`,
`TradingIntent`, `Order`, and `MarketBacktestWorld` are intentionally not
defined in the package. They show possible shapes a concrete strategy may
choose.

Read them in this order:

1. `trading_strategy_backtest.py`
   - a strategy interacts with a market world through
     `strategy.decide(...)` and `world.step(action)`
   - input/output shape: `MarketObservation -> PortfolioAction`
2. `trading_strategy_execution_intent.py`
   - a strategy output can include execution intent, not only portfolio targets
   - input/output shape: `RiskObservation -> TradingIntent`

Other concrete input/output shapes:

- `trading_strategy_orders.py`
  - input/output shape: `MarketObservation -> list[Order]`
- `trading_strategy_hold_rebalance.py`
  - input/output shape: `PortfolioObservation -> Hold | Rebalance`
- `trading_strategy_hedge_action.py`
  - input/output shape: `PortfolioRiskObservation -> HedgeAction | None`
- `trading_strategy_spread_trade.py`
  - input/output shape: `SpreadObservation -> SpreadTradeIntent | None`
- `trading_strategy_alpha_score_component.py`
  - input/output shape: `FeatureBatch -> AlphaScore`
  - this is a strategy component shape, not necessarily a full trading strategy
- `trading_strategy_stateful.py`
  - input/output shape: `PriceObservation -> PositionAction`
  - the strategy keeps internal state between calls
- `trading_strategy_execution_orders.py`
  - input/output shape: `BrokerObservation -> list[ExecutionOrder]`

In ML/RL terms, `TradingStrategy` is policy-like. The concrete observation,
action, world, and backtest shapes remain strategy- or example-specific until
multiple real use cases justify a package-level abstraction.
