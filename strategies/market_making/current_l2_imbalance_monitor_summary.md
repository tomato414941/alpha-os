# Current L2 Imbalance Monitor

This repeats the broad Hyperliquid L2 imbalance snapshot over a short window. It is a persistence check, not a fill model or trade instruction.

| asset | obs | dir | persistence | mean imbalance | mean abs imbalance | min abs imbalance | spread bps | near depth USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 3 | 1 | 1.0000 | 0.7881 | 0.7881 | 0.6984 | 0.1609 | 1522952 |
| ONDO | 3 | 1 | 1.0000 | 0.5184 | 0.5184 | 0.3427 | 1.8113 | 4925 |
| XPL | 3 | 1 | 1.0000 | 0.4912 | 0.4912 | 0.2641 | 2.2276 | 6832 |
| VVV | 3 | -1 | 1.0000 | -0.4567 | 0.4567 | 0.3465 | 2.9777 | 2741 |
| SOL | 3 | -1 | 1.0000 | -0.3658 | 0.3658 | 0.2138 | 0.1539 | 357804 |
| LIT | 3 | 1 | 1.0000 | 0.3474 | 0.3474 | 0.2705 | 0.9622 | 4244 |
| XLM | 3 | 1 | 1.0000 | 0.3286 | 0.3286 | 0.1045 | 4.2616 | 12707 |
| SUI | 3 | 1 | 1.0000 | 0.1746 | 0.1746 | 0.0782 | 0.8057 | 69152 |
| DOGE | 3 | 1 | 1.0000 | 0.1317 | 0.1317 | 0.0065 | 0.1181 | 174884 |
| ETH | 3 | -1 | 1.0000 | -0.1189 | 0.1189 | 0.0809 | 0.6143 | 10114140 |
| JTO | 3 | 1 | 1.0000 | 0.1133 | 0.1133 | 0.0048 | 7.3841 | 4316 |
| TON | 3 | 1 | 1.0000 | 0.0863 | 0.0863 | 0.0163 | 2.5433 | 15855 |
| ADA | 3 | -1 | 1.0000 | -0.0518 | 0.0518 | 0.0313 | 2.0536 | 93146 |
| NEAR | 3 | 1 | 0.6667 | 0.2592 | 0.8683 | 0.6988 | 1.5956 | 2554 |
| ZEC | 3 | -1 | 0.6667 | 0.0549 | 0.4158 | 0.0882 | 2.0728 | 54923 |
| WLD | 3 | -1 | 0.6667 | -0.3089 | 0.3834 | 0.1118 | 3.6559 | 17544 |
| HYPE | 3 | 1 | 0.6667 | 0.1790 | 0.2653 | 0.1295 | 0.1711 | 138287 |
| LTC | 3 | 1 | 0.6667 | 0.0734 | 0.1931 | 0.1156 | 2.1416 | 47184 |
| XRP | 3 | -1 | 0.6667 | 0.0563 | 0.1832 | 0.0611 | 1.1753 | 549848 |
| AERO | 3 | 1 | 0.6667 | 0.0493 | 0.0715 | 0.0334 | 3.9352 | 3677 |

## Interpretation

High persistence with high absolute imbalance is a better paper-label candidate than a one-off snapshot. It still needs 15m/1h forward labels and a real maker-fill/adverse-selection model.
