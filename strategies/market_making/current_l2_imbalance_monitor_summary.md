# Current L2 Imbalance Monitor

This repeats the broad Hyperliquid L2 imbalance snapshot over a short window. It is a persistence check, not a fill model or trade instruction.

| asset | obs | dir | persistence | mean imbalance | mean abs imbalance | min abs imbalance | spread bps | near depth USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SOL | 3 | -1 | 1.0000 | -0.3681 | 0.3681 | 0.2986 | 0.1530 | 378369 |
| BTC | 3 | 1 | 1.0000 | 0.3037 | 0.3037 | 0.0075 | 0.1599 | 4122255 |
| ETH | 3 | -1 | 1.0000 | -0.0649 | 0.0649 | 0.0145 | 0.6017 | 12285076 |
| HYPE | 3 | 1 | 0.6667 | 0.1305 | 0.1618 | 0.0469 | 0.9640 | 127355 |

## Interpretation

High persistence with high absolute imbalance is a better paper-label candidate than a one-off snapshot. It still needs 15m/1h forward labels and a real maker-fill/adverse-selection model.
