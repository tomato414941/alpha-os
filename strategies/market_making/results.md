# Market Making Results

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.market_making.hyperliquid_l2_snapshot
uv run python -m strategies.market_making.current_l2_imbalance_forward_labels
uv run python -m strategies.market_making.current_l2_imbalance_paper_gate
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

## L2 Imbalance Paper Gate

This subtracts taker round-trip fees and current spread from the book-imbalance
directional label, then checks visible 10 bps depth. It is a directional paper
gate, not a maker-fill model.

| asset | size USD | imbalance10 | cost bps | net15 bps | net1h bps | depth USD | depth usage | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HYPE | 100 | 0.2310 | 10.17 | 122.57 | 199.97 | 60570 | 0.0017 | small_paper_probe |
| HYPE | 250 | 0.2310 | 10.17 | 122.57 | 199.97 | 60570 | 0.0041 | small_paper_probe |
| HYPE | 500 | 0.2310 | 10.17 | 122.57 | 199.97 | 60570 | 0.0083 | small_paper_probe |
| HYPE | 1000 | 0.2310 | 10.17 | 122.57 | 199.97 | 60570 | 0.0165 | small_paper_probe |
| SOL | 100 | 0.4100 | 10.16 | 79.64 | 119.41 | 319875 | 0.0003 | small_paper_probe |
| SOL | 1000 | 0.4100 | 10.16 | 79.64 | 119.41 | 319875 | 0.0031 | small_paper_probe |
| BTC | 100 | 0.0167 | 10.16 | 19.41 | 18.93 | 3670441 | 0.0000 | small_paper_probe |
| ETH | 100 | -0.0338 | 10.63 | -68.00 | -115.92 | 10955176 | 0.0000 | blocked_by_cost |

Interpretation:

- `HYPE` is the strongest L2 imbalance paper candidate in this snapshot.
- `SOL` is also positive and has deeper visible 10 bps notional.
- `BTC` survives the rough gate, but its imbalance is tiny; it is less
  informative as an alpha candidate.
- This is still a directional paper gate, not proof of a maker strategy.
