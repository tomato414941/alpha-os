# Current L2 Imbalance Forward Labels

This labels whether the visible 10 bps book imbalance matched subsequent Hyperliquid price direction. It is an imbalance alpha probe, not a market-making fill model.

| asset | spread bps | imbalance10 | dir | raw 15m | dir 15m | raw 1h | dir 1h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JTO | 4.6830 | 0.2901 | 1 | 0.012475 | 0.012475 |  |  |
| XLM | 2.4802 | 0.1395 | 1 | 0.012187 | 0.012187 |  |  |
| NEAR | 1.9243 | -0.2800 | -1 | -0.011584 | 0.011584 |  |  |
| XPL | 2.1805 | 0.0082 | 1 | 0.003014 | 0.003014 |  |  |
| BNB | 1.1832 | 0.0280 | 1 | 0.002262 | 0.002262 |  |  |
| ADA | 3.0814 | 0.1401 | 1 | 0.001540 | 0.001540 |  |  |
| ETH | 0.6139 | 0.1352 | 1 | 0.001045 | 0.001045 |  |  |
| AERO | 7.8666 | 0.0632 | 1 | 0.000636 | 0.000636 |  |  |
| BTC | 0.1607 | 0.4525 | 1 | 0.000145 | 0.000145 |  |  |
| AVAX | 0.1495 | -0.3321 | -1 | 0.001524 | -0.001524 |  |  |
| SUI | 1.7469 | -0.3582 | -1 | 0.001867 | -0.001867 |  |  |
| DOGE | 2.9507 | -0.2872 | -1 | 0.001913 | -0.001913 |  |  |
| XRP | 0.8814 | -0.0966 | -1 | 0.001939 | -0.001939 |  |  |
| LTC | 6.8984 | -0.4037 | -1 | 0.003024 | -0.003024 |  |  |
| SOL | 0.1537 | -0.6055 | -1 | 0.003294 | -0.003294 |  |  |
| HYPE | 0.1711 | -0.6080 | -1 | 0.004142 | -0.004142 |  |  |
| ONDO | 2.8615 | 0.0987 | 1 | -0.004576 | -0.004576 |  |  |
| ENA | 2.6590 | -0.4834 | -1 | 0.005524 | -0.005524 |  |  |
| ZEC | 0.2400 | -0.7326 | -1 | 0.006482 | -0.006482 |  |  |
| TON | 2.3599 | -0.0212 | -1 | 0.006863 | -0.006863 |  |  |

## Interpretation

Positive directional return means the snapshot's visible imbalance pointed in the right price direction. A market-making strategy still needs queue position, fill probability, maker/taker fees, and adverse selection estimates.
