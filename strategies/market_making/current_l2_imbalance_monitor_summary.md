# Current L2 Imbalance Monitor

This repeats the broad Hyperliquid L2 imbalance snapshot over a short window. It is a persistence check, not a fill model or trade instruction.

| asset | obs | dir | persistence | mean imbalance | mean abs imbalance | min abs imbalance | spread bps | near depth USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JTO | 3 | -1 | 1.0000 | -0.8916 | 0.8916 | 0.8675 | 7.0095 | 1876 |
| HYPE | 3 | -1 | 1.0000 | -0.6768 | 0.6768 | 0.6578 | 0.3152 | 182454 |
| LIT | 3 | 1 | 1.0000 | 0.5903 | 0.5903 | 0.4484 | 4.9999 | 4655 |
| BTC | 3 | 1 | 1.0000 | 0.5321 | 0.5321 | 0.4582 | 0.1578 | 3019455 |
| WLD | 3 | -1 | 1.0000 | -0.4699 | 0.4699 | 0.4164 | 7.4499 | 17455 |
| NEAR | 3 | 1 | 1.0000 | 0.4216 | 0.4216 | 0.0906 | 2.2734 | 16666 |
| ZEC | 3 | -1 | 1.0000 | -0.3296 | 0.3296 | 0.0970 | 1.1749 | 68103 |
| ENA | 3 | -1 | 1.0000 | -0.2986 | 0.2986 | 0.1663 | 2.0573 | 29494 |
| LTC | 3 | -1 | 1.0000 | -0.2640 | 0.2640 | 0.2441 | 2.3798 | 47454 |
| AERO | 3 | 1 | 1.0000 | 0.2609 | 0.2609 | 0.1848 | 5.1239 | 3776 |
| XMR | 3 | 1 | 1.0000 | 0.2308 | 0.2308 | 0.0751 | 1.0559 | 14784 |
| ONDO | 3 | -1 | 1.0000 | -0.2230 | 0.2230 | 0.0331 | 3.5366 | 13634 |
| XRP | 3 | -1 | 1.0000 | -0.2080 | 0.2080 | 0.1594 | 0.8511 | 593877 |
| SUI | 3 | -1 | 1.0000 | -0.1561 | 0.1561 | 0.0564 | 1.2273 | 70815 |
| VVV | 3 | -1 | 0.6667 | -0.3713 | 0.4367 | 0.0981 | 2.4896 | 2085 |
| TON | 3 | 1 | 0.6667 | 0.2503 | 0.3773 | 0.1904 | 1.6867 | 18117 |
| SOL | 3 | 1 | 0.6667 | 0.1924 | 0.3063 | 0.1708 | 0.1483 | 322735 |
| BNB | 3 | -1 | 0.6667 | -0.0742 | 0.1690 | 0.1421 | 1.9211 | 68201 |
| DOGE | 3 | 1 | 0.6667 | 0.1191 | 0.1565 | 0.0560 | 0.4994 | 131824 |
| ADA | 3 | 1 | 0.6667 | 0.0367 | 0.0834 | 0.0610 | 2.9124 | 85198 |

## Interpretation

High persistence with high absolute imbalance is a better paper-label candidate than a one-off snapshot. It still needs 15m/1h forward labels and a real maker-fill/adverse-selection model.
