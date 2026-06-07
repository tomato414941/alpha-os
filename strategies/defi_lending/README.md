# DeFi Lending

This lane screens lending and borrowing markets as funding-pressure sources.

It is separate from `defi_yield`: yield pools can be generic carry products,
while lending markets expose borrow demand, utilization, available liquidity,
and collateral-specific stress.

## Commands

```bash
uv run python -m strategies.defi_lending.current_morpho_lending_rates
```

## Current Status

This is a current snapshot screen only. It does not yet model liquidation risk,
oracle risk, collateral drawdown, withdrawal queues, rate persistence, or gas
costs.

