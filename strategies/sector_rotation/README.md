# Sector Rotation

This lane looks for crypto category rotation using public CoinGecko category
snapshots.

It is not a trading strategy yet. The first probe ranks categories by current
24h market-cap change, scale, and concentration proxy.
The second probe maps top category constituents to Hyperliquid-tradable symbols
and labels short forward returns.
The third probe joins those category candidates to current perp funding and
liquidity context.

## Commands

```bash
uv run python -m strategies.sector_rotation.current_coingecko_category_rotation
uv run python -m strategies.sector_rotation.current_category_tradable_forward_labels
uv run python -m strategies.sector_rotation.current_category_perp_context
```
