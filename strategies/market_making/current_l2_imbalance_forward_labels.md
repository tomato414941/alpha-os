# Current L2 Imbalance Forward Labels

This labels whether the visible 10 bps book imbalance matched subsequent Hyperliquid price direction. It is an imbalance alpha probe, not a market-making fill model.

| asset | spread bps | imbalance10 | dir | raw 15m | dir 15m | raw 1h | dir 1h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 0.1599 | 0.6875 | 1 |  |  |  |  |
| ETH | 0.6016 | 0.0231 | 1 |  |  |  |  |
| SOL | 0.1529 | -0.3960 | -1 |  |  |  |  |
| HYPE | 0.1606 | 0.1703 | 1 |  |  |  |  |

## Interpretation

Positive directional return means the snapshot's visible imbalance pointed in the right price direction. A market-making strategy still needs queue position, fill probability, maker/taker fees, and adverse selection estimates.
