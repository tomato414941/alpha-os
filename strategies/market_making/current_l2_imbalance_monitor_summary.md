# Current L2 Imbalance Monitor

This repeats the broad Hyperliquid L2 imbalance snapshot over a short window. It is a persistence check, not a fill model or trade instruction.

| asset | obs | dir | persistence | mean imbalance | mean abs imbalance | min abs imbalance | spread bps | near depth USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AERO | 3 | 1 | 1.0000 | 0.7258 | 0.7258 | 0.7128 | 10.9252 | 308 |
| HYPE | 3 | -1 | 1.0000 | -0.7006 | 0.7006 | 0.5339 | 0.4618 | 153347 |
| LIT | 3 | -1 | 1.0000 | -0.6176 | 0.6176 | 0.3608 | 8.1279 | 2198 |
| VVV | 3 | 1 | 1.0000 | 0.4445 | 0.4445 | 0.3450 | 0.5495 | 7636 |
| ONDO | 3 | -1 | 1.0000 | -0.3157 | 0.3157 | 0.1353 | 1.6283 | 10713 |
| SOL | 3 | 1 | 1.0000 | 0.2780 | 0.2780 | 0.0349 | 0.1476 | 400075 |
| BTC | 3 | -1 | 1.0000 | -0.2654 | 0.2654 | 0.1164 | 0.1558 | 4641067 |
| AVAX | 3 | -1 | 1.0000 | -0.1500 | 0.1500 | 0.0529 | 1.2133 | 55286 |
| XRP | 3 | 1 | 1.0000 | 0.1367 | 0.1367 | 0.0992 | 0.8451 | 554204 |
| SUI | 3 | 1 | 1.0000 | 0.1353 | 0.1353 | 0.0415 | 1.8145 | 97466 |
| DOGE | 3 | 1 | 1.0000 | 0.1295 | 0.1295 | 0.0402 | 1.1841 | 161968 |
| TON | 3 | 1 | 0.6667 | 0.2005 | 0.7550 | 0.4637 | 1.1317 | 10643 |
| WLD | 3 | 1 | 0.6667 | 0.6228 | 0.6403 | 0.0263 | 4.7122 | 14479 |
| LTC | 3 | 1 | 0.6667 | 0.2219 | 0.3396 | 0.1765 | 0.9968 | 45183 |
| JTO | 3 | 1 | 0.6667 | -0.1854 | 0.2519 | 0.0183 | 12.1414 | 2249 |
| ZEC | 3 | -1 | 0.6667 | -0.2223 | 0.2359 | 0.0204 | 1.1301 | 89409 |
| ADA | 3 | 1 | 0.6667 | -0.0271 | 0.2146 | 0.0606 | 2.7306 | 55649 |
| ETH | 3 | -1 | 0.6667 | 0.0419 | 0.1641 | 0.0828 | 0.5871 | 8922293 |
| NEAR | 3 | -1 | 0.6667 | 0.0082 | 0.1377 | 0.0676 | 4.7189 | 29694 |
| XMR | 3 | 1 | 0.6667 | 0.1206 | 0.1268 | 0.0093 | 1.4700 | 10408 |

## Interpretation

High persistence with high absolute imbalance is a better paper-label candidate than a one-off snapshot. It still needs 15m/1h forward labels and a real maker-fill/adverse-selection model.
