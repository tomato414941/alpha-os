# Market Making

This lane looks for spread-capture and inventory-control opportunities.

The first probe uses Hyperliquid L2 order-book snapshots for liquid perps and
records:

- top-of-book spread
- depth within 10 bps and 50 bps
- near-book imbalance
- top-level order counts

## Commands

```bash
uv run python -m strategies.market_making.hyperliquid_l2_snapshot
```

## Current Status

This is not yet a market-making strategy. It only confirms that order-book
snapshots are reachable and starts measuring whether there is enough spread and
depth to justify a fill model.

