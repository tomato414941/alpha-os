# Current L2 Imbalance Monitor

This repeats the broad Hyperliquid L2 imbalance snapshot over a short window. It is a persistence check, not a fill model or trade instruction.

| asset | obs | dir | persistence | mean imbalance | mean abs imbalance | min abs imbalance | spread bps | near depth USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WLD | 3 | 1 | 1.0000 | 0.6003 | 0.6003 | 0.2385 | 4.8235 | 11616 |
| ENA | 3 | -1 | 1.0000 | -0.5633 | 0.5633 | 0.3423 | 0.9343 | 16095 |
| XMR | 3 | 1 | 1.0000 | 0.5322 | 0.5322 | 0.4232 | 1.7594 | 6497 |
| JTO | 3 | -1 | 1.0000 | -0.4534 | 0.4534 | 0.2547 | 6.7101 | 2961 |
| ADA | 3 | -1 | 1.0000 | -0.3642 | 0.3642 | 0.3300 | 1.6171 | 51478 |
| LTC | 3 | -1 | 1.0000 | -0.2617 | 0.2617 | 0.1299 | 1.4814 | 34745 |
| BNB | 3 | 1 | 1.0000 | 0.1642 | 0.1642 | 0.0791 | 0.9941 | 95991 |
| AVAX | 3 | -1 | 1.0000 | -0.1086 | 0.1086 | 0.0214 | 1.6702 | 49715 |
| LIT | 3 | 1 | 0.6667 | -0.0002 | 0.6367 | 0.2787 | 6.3137 | 2606 |
| NEAR | 3 | 1 | 0.6667 | 0.2571 | 0.4599 | 0.3042 | 2.1310 | 16155 |
| ZEC | 3 | -1 | 0.6667 | 0.0644 | 0.4170 | 0.1948 | 1.6809 | 52214 |
| TON | 3 | 1 | 0.6667 | 0.3531 | 0.3850 | 0.0477 | 2.7354 | 26333 |
| BTC | 3 | -1 | 0.6667 | 0.0369 | 0.3525 | 0.0290 | 0.2639 | 3032100 |
| SOL | 3 | -1 | 0.6667 | -0.1556 | 0.3426 | 0.2335 | 0.2519 | 366591 |
| ONDO | 3 | 1 | 0.6667 | -0.0337 | 0.2126 | 0.0615 | 1.9253 | 11140 |
| AERO | 3 | -1 | 0.6667 | -0.1947 | 0.2045 | 0.0146 | 5.2693 | 2834 |
| SUI | 3 | 1 | 0.6667 | 0.0883 | 0.1975 | 0.0070 | 1.6289 | 49777 |
| HYPE | 3 | -1 | 0.6667 | -0.1476 | 0.1932 | 0.0684 | 0.5030 | 135824 |
| ETH | 3 | 1 | 0.6667 | 0.0577 | 0.1459 | 0.1010 | 0.7920 | 9086544 |
| XRP | 3 | 1 | 0.6667 | 0.1408 | 0.1453 | 0.0068 | 1.1575 | 498024 |

## Interpretation

High persistence with high absolute imbalance is a better paper-label candidate than a one-off snapshot. It still needs 15m/1h forward labels and a real maker-fill/adverse-selection model.
