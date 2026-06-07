# Current L2 Imbalance Forward Labels

This labels whether the visible 10 bps book imbalance matched subsequent Hyperliquid price direction. It is an imbalance alpha probe, not a market-making fill model.

| asset | spread bps | imbalance10 | dir | raw 15m | dir 15m | raw 1h | dir 1h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HYPE | 0.1727 | 0.2310 | 1 | 0.013274 | 0.013274 | 0.021014 | 0.021014 |
| SOL | 0.1573 | 0.4100 | 1 | 0.008980 | 0.008980 | 0.012956 | 0.012956 |
| BTC | 0.1623 | 0.0167 | 1 | 0.002958 | 0.002958 | 0.002909 | 0.002909 |
| ETH | 0.6291 | -0.0338 | -1 | 0.005737 | -0.005737 | 0.010529 | -0.010529 |

## Interpretation

Positive directional return means the snapshot's visible imbalance pointed in the right price direction. A market-making strategy still needs queue position, fill probability, maker/taker fees, and adverse selection estimates.
