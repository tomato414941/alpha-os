# Market Making Results

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.market_making.hyperliquid_l2_snapshot
uv run python -m strategies.market_making.current_l2_imbalance_forward_labels
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

## L2 Imbalance Forward Labels

This labels whether the visible 10 bps book imbalance matched subsequent
Hyperliquid price direction. It is an imbalance alpha probe, not a market-making
fill model.

| asset | spread bps | imbalance10 | dir | raw 15m | dir 15m | raw 1h | dir 1h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HYPE | 0.1727 | 0.2310 | 1 | 0.013274 | 0.013274 | 0.021014 | 0.021014 |
| SOL | 0.1573 | 0.4100 | 1 | 0.008980 | 0.008980 | 0.012956 | 0.012956 |
| BTC | 0.1623 | 0.0167 | 1 | 0.002958 | 0.002958 | 0.002909 | 0.002909 |
| ETH | 0.6291 | -0.0338 | -1 | 0.005737 | -0.005737 | 0.010529 | -0.010529 |

Interpretation:

- `HYPE` and `SOL` had strong positive imbalance labels over both 15m and 1h.
- `ETH` is a useful negative example: its visible imbalance pointed down, but
  price moved up.
- This suggests near-book imbalance may be an inventory alpha input, but it is
  not yet a market-making strategy without fill and adverse-selection modeling.
