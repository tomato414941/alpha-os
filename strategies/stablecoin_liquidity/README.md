# Stablecoin Liquidity

This lane treats stablecoin supply and peg state as market liquidity context.

The first probe uses DeFiLlama stablecoin data and records:

- current circulating supply
- day, week, and month supply changes
- peg type and mechanism
- current price when available

The peg-stress screen turns the same source into depeg/repeg and premium
mean-reversion watches. It does not assume the price is tradable; redemption
route, venue depth, custody, and repeated snapshots are required before paper
action.

## Commands

```bash
uv run python -m strategies.stablecoin_liquidity.current_supply_snapshot
uv run python -m strategies.stablecoin_liquidity.current_chain_stablecoin_migration
uv run python -m strategies.stablecoin_liquidity.current_supply_market_forward_labels
uv run python -m strategies.stablecoin_liquidity.current_peg_stress_screen
```

## Current Status

This is not a direct trading strategy. It is a liquidity and stress context
probe. The chain migration screen is a stablecoin distribution proxy, not a
bridge-fill feed. The next useful step is joining stablecoin supply and chain
migration changes to crypto returns, DeFi yield, funding, and risk-off regimes.
