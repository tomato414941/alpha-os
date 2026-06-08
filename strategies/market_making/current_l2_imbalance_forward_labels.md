# Current L2 Imbalance Forward Labels

This labels whether the visible 10 bps book imbalance matched subsequent Hyperliquid price direction. It is an imbalance alpha probe, not a market-making fill model.

| asset | spread bps | imbalance10 | dir | raw 15m | dir 15m | raw 1h | dir 1h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 0.1583 | 0.7403 | 1 |  |  |  |  |
| ETH | 0.5939 | 0.1378 | 1 |  |  |  |  |
| SOL | 0.1510 | -0.5921 | -1 |  |  |  |  |
| HYPE | 0.1676 | -0.3833 | -1 |  |  |  |  |
| WLD | 6.7975 | 0.2198 | 1 |  |  |  |  |
| JTO | 5.1844 | -0.0957 | -1 |  |  |  |  |
| ONDO | 0.8658 | -0.1206 | -1 |  |  |  |  |
| AERO | 6.2603 | -0.3430 | -1 |  |  |  |  |
| ZEC | 1.1457 | -0.1615 | -1 |  |  |  |  |
| NEAR | 0.4913 | -0.2284 | -1 |  |  |  |  |
| DOGE | 1.0460 | 0.0843 | 1 |  |  |  |  |
| LTC | 0.7014 | -0.1949 | -1 |  |  |  |  |
| XRP | 1.7357 | -0.0873 | -1 |  |  |  |  |
| SUI | 1.9804 | 0.3281 | 1 |  |  |  |  |
| TON | 5.2763 | 0.0829 | 1 |  |  |  |  |
| LIT | 8.3887 | 0.7638 | 1 |  |  |  |  |
| VVV | 1.8380 | -0.0789 | -1 |  |  |  |  |
| ADA | 1.8176 | -0.0752 | -1 |  |  |  |  |
| AVAX | 1.9146 | -0.1627 | -1 |  |  |  |  |
| BNB | 1.3251 | 0.2674 | 1 |  |  |  |  |

## Interpretation

Positive directional return means the snapshot's visible imbalance pointed in the right price direction. A market-making strategy still needs queue position, fill probability, maker/taker fees, and adverse selection estimates.
