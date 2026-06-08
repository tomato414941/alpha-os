# DeFi Lending

This lane screens lending and borrowing markets as funding-pressure sources.

It is separate from `defi_yield`: yield pools can be generic carry products,
while lending markets expose borrow demand, utilization, available liquidity,
and collateral-specific stress.

## Commands

```bash
uv run python -m strategies.defi_lending.current_morpho_lending_rates
uv run python -m strategies.defi_lending.current_lending_stress_actionability
```

## Current Status

This is current snapshot work only. The actionability check separates visible
remaining lending capacity from no-liquidity stress, but it still does not model
liquidation risk, oracle risk, collateral drawdown, withdrawal queues, rate
persistence, or gas costs.
