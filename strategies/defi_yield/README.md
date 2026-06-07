# DeFi Yield

This lane screens non-price yield opportunities. It is separate from directional
price prediction and portfolio rotation.

The first screen uses DeFiLlama pools and filters for stablecoin-like, single
exposure pools without impermanent-loss risk flags.

## Commands

```bash
uv run python -m strategies.defi_yield.current_yield_screen
uv run python -m strategies.defi_yield.current_yield_quality_screen
```

## Current Status

This is a yield-source inventory, not a trading recommendation. The screen does
now separates base APY from reward-heavy APY, but it does not yet model:

- smart-contract risk
- stablecoin depeg risk
- withdrawal or liquidity constraints
- gas and bridge costs
- custody and operational risk
- APY decay after capital enters the pool
