# Current L2 Imbalance Monitor

This repeats the broad Hyperliquid L2 imbalance snapshot over a short window. It is a persistence check, not a fill model or trade instruction.

| asset | obs | dir | persistence | mean imbalance | mean abs imbalance | min abs imbalance | spread bps | near depth USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NEAR | 3 | 1 | 1.0000 | 0.5974 | 0.5974 | 0.5251 | 1.4081 | 82839 |
| LTC | 3 | -1 | 1.0000 | -0.4144 | 0.4144 | 0.3884 | 2.0059 | 37977 |
| ENA | 3 | -1 | 1.0000 | -0.3857 | 0.3857 | 0.3165 | 0.8469 | 34430 |
| ONDO | 3 | 1 | 1.0000 | 0.3530 | 0.3530 | 0.0436 | 8.5783 | 5519 |
| TON | 3 | 1 | 1.0000 | 0.3014 | 0.3014 | 0.0301 | 1.5502 | 69745 |
| JTO | 3 | -1 | 1.0000 | -0.2982 | 0.2982 | 0.2091 | 7.8175 | 1364 |
| AVAX | 3 | 1 | 1.0000 | 0.2427 | 0.2427 | 0.0595 | 1.3271 | 53874 |
| ETH | 3 | -1 | 1.0000 | -0.2011 | 0.2011 | 0.0675 | 0.5883 | 7152510 |
| BNB | 3 | 1 | 1.0000 | 0.0400 | 0.0400 | 0.0043 | 2.0411 | 100771 |
| WLD | 3 | -1 | 0.6667 | -0.2833 | 0.5725 | 0.4338 | 2.5516 | 16522 |
| ZEC | 3 | -1 | 0.6667 | -0.0256 | 0.5202 | 0.3739 | 1.3850 | 51849 |
| XMR | 3 | -1 | 0.6667 | 0.0336 | 0.4614 | 0.1827 | 0.3142 | 8476 |
| HYPE | 3 | 1 | 0.6667 | 0.3918 | 0.4350 | 0.0648 | 0.8414 | 134215 |
| AERO | 3 | 1 | 0.6667 | 0.1936 | 0.2796 | 0.1290 | 2.6250 | 3482 |
| BTC | 3 | 1 | 0.6667 | 0.2177 | 0.2698 | 0.0782 | 0.1581 | 4684934 |
| LIT | 3 | 1 | 0.6667 | 0.0045 | 0.1912 | 0.0253 | 6.3775 | 4496 |
| SOL | 3 | 1 | 0.6667 | 0.1854 | 0.1855 | 0.0001 | 0.1494 | 559754 |
| VVV | 3 | -1 | 0.6667 | -0.1631 | 0.1710 | 0.0118 | 7.0194 | 3474 |
| XRP | 3 | -1 | 0.6667 | -0.1049 | 0.1522 | 0.0710 | 0.8524 | 542084 |
| SUI | 3 | -1 | 0.6667 | -0.0924 | 0.1107 | 0.0207 | 1.4525 | 82082 |

## Interpretation

High persistence with high absolute imbalance is a better paper-label candidate than a one-off snapshot. It still needs 15m/1h forward labels and a real maker-fill/adverse-selection model.
