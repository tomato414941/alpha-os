# Current L2 Imbalance Monitor

This repeats the broad Hyperliquid L2 imbalance snapshot over a short window. It is a persistence check, not a fill model or trade instruction.

| asset | obs | dir | persistence | mean imbalance | mean abs imbalance | min abs imbalance | spread bps | near depth USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WLD | 3 | 1 | 1.0000 | 0.6955 | 0.6955 | 0.5364 | 5.1155 | 6015 |
| AERO | 3 | 1 | 1.0000 | 0.6799 | 0.6799 | 0.5699 | 10.2842 | 946 |
| SOL | 3 | 1 | 1.0000 | 0.6095 | 0.6095 | 0.3795 | 0.2999 | 265221 |
| ONDO | 3 | 1 | 1.0000 | 0.5681 | 0.5681 | 0.5146 | 7.1113 | 5130 |
| LIT | 3 | -1 | 1.0000 | -0.5159 | 0.5159 | 0.1090 | 7.6675 | 2881 |
| SUI | 3 | 1 | 1.0000 | 0.4505 | 0.4505 | 0.2366 | 2.4979 | 42911 |
| JTO | 3 | -1 | 1.0000 | -0.3619 | 0.3619 | 0.1588 | 7.7754 | 3073 |
| XRP | 3 | 1 | 1.0000 | 0.2819 | 0.2819 | 0.2698 | 0.8645 | 442137 |
| AVAX | 3 | 1 | 1.0000 | 0.1846 | 0.1846 | 0.0286 | 2.0155 | 37251 |
| TON | 3 | -1 | 1.0000 | -0.1581 | 0.1581 | 0.0206 | 3.4640 | 30563 |
| DOGE | 3 | -1 | 1.0000 | -0.1154 | 0.1154 | 0.0329 | 1.2713 | 163881 |
| XMR | 3 | 1 | 0.6667 | 0.1981 | 0.4537 | 0.3087 | 3.2936 | 6960 |
| ENA | 3 | 1 | 0.6667 | 0.0674 | 0.3659 | 0.1666 | 5.5249 | 25575 |
| NEAR | 3 | -1 | 0.6667 | -0.1587 | 0.3608 | 0.1623 | 3.0900 | 28349 |
| HYPE | 3 | 1 | 0.6667 | 0.1526 | 0.3413 | 0.1547 | 0.8062 | 101673 |
| VVV | 3 | -1 | 0.6667 | -0.0265 | 0.2601 | 0.0310 | 5.8059 | 3110 |
| ADA | 3 | -1 | 0.6667 | 0.0512 | 0.2362 | 0.1095 | 2.5995 | 56214 |
| BTC | 3 | -1 | 0.6667 | -0.1431 | 0.2185 | 0.1132 | 0.2104 | 2951384 |
| LTC | 3 | 1 | 0.6667 | 0.0597 | 0.1354 | 0.0775 | 1.6232 | 45046 |
| BNB | 3 | -1 | 0.6667 | -0.0881 | 0.1070 | 0.0284 | 2.7191 | 109313 |

## Interpretation

High persistence with high absolute imbalance is a better paper-label candidate than a one-off snapshot. It still needs 15m/1h forward labels and a real maker-fill/adverse-selection model.
