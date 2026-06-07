# Perp Market Map

This lane maps perpetual futures markets before turning anything into a
strategy.

The first probe uses Hyperliquid public market contexts:

- current funding
- open interest
- 24h notional volume
- premium
- mark/oracle dislocation
- impact spread

## Commands

```bash
uv run python -m strategies.perp_market_map.current_hyperliquid_snapshot
```

## Current Status

This is not a trading strategy. It is a market map for finding where carry,
crowding, dislocation, and liquidity might justify deeper work.

