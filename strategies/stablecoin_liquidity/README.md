# Stablecoin Liquidity

This lane treats stablecoin supply and peg state as market liquidity context.

The first probe uses DeFiLlama stablecoin data and records:

- current circulating supply
- day, week, and month supply changes
- peg type and mechanism
- current price when available

## Commands

```bash
uv run python -m strategies.stablecoin_liquidity.current_supply_snapshot
```

## Current Status

This is not a direct trading strategy. It is a liquidity and stress context
probe. The next useful step is joining stablecoin supply changes to crypto
returns, DeFi yield, funding, and risk-off regimes.

