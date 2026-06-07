# Market Making Results

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.market_making.hyperliquid_l2_snapshot
```

Interpretation:

- low spread means spread capture is harder unless maker rebates or queue edge
  exist
- shallow 10 bps depth limits capacity
- imbalance can become a short-horizon signal only after repeated snapshots
- this snapshot does not model queue position, fill probability, adverse
  selection, or fees

## Snapshot

| asset | spread bps | bid depth 10 bps | ask depth 10 bps | imbalance 10 bps |
| --- | ---: | ---: | ---: | ---: |
| BTC | 0.1623 | 61.6031 | 59.5817 | 0.0167 |
| ETH | 0.6291 | 6891.5645 | 7373.7756 | -0.0338 |
| SOL | 0.1573 | 12025.7400 | 5032.0500 | 0.4100 |
| HYPE | 0.1727 | 1674.6500 | 1046.1700 | 0.2310 |

The visible spread is small on liquid assets, so a real market-making strategy
would need queue edge, rebates, or inventory alpha. SOL and HYPE show one-sided
near-book depth in this snapshot, but that is only useful if it persists and can
be connected to fills or short-horizon price movement.

