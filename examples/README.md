# Examples

These examples are concrete sketches, not alpha-os public API.

The maintained package currently exposes only the `TradingStrategy` contract:

```text
decide(input) -> output
```

The example-local names such as `MarketObservation`, `PortfolioAction`,
`TradingIntent`, and `MarketBacktestWorld` are intentionally not defined in the
package. They show possible shapes a concrete strategy may choose.

Read them in this order:

1. `trading_strategy_rollout.py`
   - minimal `observation -> strategy -> action` flow
2. `trading_strategy_execution_intent.py`
   - a strategy output can include execution intent, not only portfolio targets
3. `trading_strategy_backtest.py`
   - a small RL-shaped rollout with `strategy.decide(...)` and
     `world.step(action)`

In ML/RL terms, `TradingStrategy` is policy-like. The concrete observation,
action, world, and backtest shapes remain strategy- or example-specific until
multiple real use cases justify a package-level abstraction.
