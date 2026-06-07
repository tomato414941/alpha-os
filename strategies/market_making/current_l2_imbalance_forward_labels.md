# Current L2 Imbalance Forward Labels

This labels whether the visible 10 bps book imbalance matched subsequent Hyperliquid price direction. It is an imbalance alpha probe, not a market-making fill model.

| asset | spread bps | imbalance10 | dir | raw 15m | dir 15m | raw 1h | dir 1h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 0.1607 | 0.4525 | 1 |  |  |  |  |
| ETH | 0.6139 | 0.1352 | 1 |  |  |  |  |
| SOL | 0.1537 | -0.6055 | -1 |  |  |  |  |
| HYPE | 0.1711 | -0.6080 | -1 |  |  |  |  |
| WLD | 1.2468 | -0.7746 | -1 |  |  |  |  |
| JTO | 4.6830 | 0.2901 | 1 |  |  |  |  |
| ONDO | 2.8615 | 0.0987 | 1 |  |  |  |  |
| AERO | 7.8666 | 0.0632 | 1 |  |  |  |  |
| ZEC | 0.2400 | -0.7326 | -1 |  |  |  |  |
| NEAR | 1.9243 | -0.2800 | -1 |  |  |  |  |
| DOGE | 2.9507 | -0.2872 | -1 |  |  |  |  |
| LTC | 6.8984 | -0.4037 | -1 |  |  |  |  |
| XRP | 0.8814 | -0.0966 | -1 |  |  |  |  |
| SUI | 1.7469 | -0.3582 | -1 |  |  |  |  |
| TON | 2.3599 | -0.0212 | -1 |  |  |  |  |
| LIT | 10.2078 | -0.4180 | -1 |  |  |  |  |
| VVV | 3.5674 | 0.2179 | 1 |  |  |  |  |
| ADA | 3.0814 | 0.1401 | 1 |  |  |  |  |
| AVAX | 0.1495 | -0.3321 | -1 |  |  |  |  |
| XLM | 2.4802 | 0.1395 | 1 |  |  |  |  |

## Interpretation

Positive directional return means the snapshot's visible imbalance pointed in the right price direction. A market-making strategy still needs queue position, fill probability, maker/taker fees, and adverse selection estimates.
