# DEX Pool Flow

This lane looks at DEX pool activity: liquidity, volume, buy/sell pressure, and
short-horizon price movement.

It is separate from centralized perp funding and from CoinGecko trending
attention. The first screen uses GeckoTerminal public trending pools.

## Commands

```bash
uv run python -m strategies.dex_pool_flow.current_geckoterminal_pool_flow
```

## Current Status

This is a pool-flow candidate screen, not a trade instruction. Pool data can be
thin, manipulated, delayed, or hard to execute. Any candidate needs route
simulation, pool depth, slippage, gas, MEV, token-transfer restrictions, and
contract-risk checks.
