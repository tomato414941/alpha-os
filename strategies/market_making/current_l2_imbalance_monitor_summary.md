# Current L2 Imbalance Monitor

This repeats the broad Hyperliquid L2 imbalance snapshot over a short window. It is a persistence check, not a fill model or trade instruction.

| asset | obs | dir | persistence | mean imbalance | mean abs imbalance | min abs imbalance | spread bps | near depth USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NEAR | 3 | 1 | 1.0000 | 0.6099 | 0.6099 | 0.2167 | 7.1825 | 13650 |
| SOL | 3 | -1 | 1.0000 | -0.2353 | 0.2353 | 0.1875 | 0.2474 | 447876 |
| AERO | 3 | 1 | 1.0000 | 0.2094 | 0.2094 | 0.0283 | 7.3602 | 968 |
| ADA | 3 | -1 | 1.0000 | -0.1955 | 0.1955 | 0.0231 | 0.9876 | 68079 |
| TON | 3 | 1 | 1.0000 | 0.1466 | 0.1466 | 0.1278 | 2.1165 | 31132 |
| ETH | 3 | 1 | 1.0000 | 0.1452 | 0.1452 | 0.1241 | 1.3772 | 10146592 |
| XRP | 3 | 1 | 1.0000 | 0.0798 | 0.0798 | 0.0298 | 1.1375 | 653189 |
| ENA | 3 | -1 | 1.0000 | -0.0701 | 0.0701 | 0.0252 | 4.0737 | 33496 |
| WLD | 3 | -1 | 0.6667 | -0.1095 | 0.7150 | 0.2691 | 1.8072 | 10470 |
| XMR | 3 | -1 | 0.6667 | -0.0948 | 0.5937 | 0.4172 | 4.5203 | 6618 |
| JTO | 3 | 1 | 0.6667 | 0.3060 | 0.5365 | 0.3457 | 12.3460 | 681 |
| LIT | 3 | 1 | 0.6667 | 0.3477 | 0.4576 | 0.1649 | 4.3067 | 5141 |
| BTC | 3 | 1 | 0.6667 | 0.0017 | 0.3916 | 0.0247 | 0.1563 | 3049187 |
| HYPE | 3 | -1 | 0.6667 | -0.0255 | 0.3146 | 0.1000 | 0.1551 | 158599 |
| ZEC | 3 | -1 | 0.6667 | -0.1048 | 0.2712 | 0.2495 | 2.2582 | 87825 |
| ONDO | 3 | -1 | 0.6667 | -0.1098 | 0.2506 | 0.0200 | 2.8114 | 9718 |
| VVV | 3 | 1 | 0.6667 | -0.0302 | 0.2053 | 0.0057 | 3.1067 | 6321 |
| LTC | 3 | -1 | 0.6667 | -0.0452 | 0.1522 | 0.1011 | 2.3853 | 47387 |
| BNB | 3 | 1 | 0.6667 | 0.1000 | 0.1303 | 0.0288 | 1.3800 | 118596 |
| DOGE | 3 | 1 | 0.6667 | 0.0941 | 0.0991 | 0.0075 | 1.3060 | 161604 |

## Interpretation

High persistence with high absolute imbalance is a better paper-label candidate than a one-off snapshot. It still needs 15m/1h forward labels and a real maker-fill/adverse-selection model.
