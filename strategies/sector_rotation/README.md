# Sector Rotation

This lane looks for crypto category rotation using public CoinGecko category
snapshots.

It is not a trading strategy yet. The first probe ranks categories by current
24h market-cap change, scale, and concentration proxy.

## Commands

```bash
uv run python -m strategies.sector_rotation.current_coingecko_category_rotation
```
