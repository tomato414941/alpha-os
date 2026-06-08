# Current Microstructure Flow Snapshot

This joins Hyperliquid public book imbalance with recent trade-print imbalance. It is a short-horizon microstructure observation, not a deployable market-making model.

| asset | action | dir | pressure | book imb | trade imb | trades | buy USD | sell USD | spread bps | depth 10bps USD | window s | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| OP | aligned_pressure_watch | -1 | -0.8124 | -0.6248 | -1.0000 | 10 | 0 | 1078 | 4.1110 | 440 | 0 | book imbalance and taker flow point the same way |
| MON | aligned_pressure_watch | -1 | -0.5658 | -0.6503 | -0.4813 | 10 | 137 | 392 | 2.3130 | 1769 | 189 | book imbalance and taker flow point the same way |
| HYPE | aligned_pressure_watch | 1 | 0.5514 | 0.3390 | 0.7637 | 10 | 2670 | 358 | 0.1577 | 86446 | 5 | book imbalance and taker flow point the same way |
| POL | aligned_pressure_watch | 1 | 0.5147 | 0.1705 | 0.8589 | 10 | 1118 | 85 | 2.6686 | 6365 | 231 | book imbalance and taker flow point the same way |
| CHIP | aligned_pressure_watch | 1 | 0.4422 | 0.0133 | 0.8712 | 10 | 2453 | 169 | 3.5827 | 7557 | 333 | book imbalance and taker flow point the same way |
| BTC | aligned_pressure_watch | 1 | 0.4283 | 0.3480 | 0.5087 | 10 | 387 | 126 | 0.1578 | 3561930 | 4 | book imbalance and taker flow point the same way |
| BERA | aligned_pressure_watch | 1 | 0.3171 | 0.1517 | 0.4825 | 10 | 467 | 163 | 6.3314 | 3938 | 300 | book imbalance and taker flow point the same way |
| BNB | aligned_pressure_watch | 1 | 0.2534 | 0.0700 | 0.4368 | 10 | 858 | 336 | 1.4824 | 97028 | 48 | book imbalance and taker flow point the same way |
| ARB | book_trade_divergence_watch | -1 | -0.4645 | 0.0709 | -1.0000 | 10 | 0 | 1269 | 2.4036 | 45748 | 76 | book imbalance and taker flow disagree |
| ETH | book_trade_divergence_watch | -1 | -0.4235 | 0.1431 | -0.9900 | 10 | 100 | 19940 | 0.5936 | 10650019 | 7 | book imbalance and taker flow disagree |
| SUI | book_trade_divergence_watch | 1 | 0.4164 | -0.0967 | 0.9294 | 10 | 682 | 25 | 0.5262 | 63461 | 70 | book imbalance and taker flow disagree |
| MEGA | book_trade_divergence_watch | -1 | -0.2428 | 0.2373 | -0.7228 | 10 | 203 | 1264 | 4.3566 | 3644 | 156 | book imbalance and taker flow disagree |
| SEI | book_trade_divergence_watch | 1 | 0.2013 | -0.2894 | 0.6920 | 10 | 1475 | 269 | 4.8231 | 7895 | 147 | book imbalance and taker flow disagree |
| APT | book_trade_divergence_watch | 1 | 0.1326 | 0.6377 | -0.3725 | 10 | 393 | 859 | 4.4069 | 6799 | 147 | book imbalance and taker flow disagree |
| SOL | book_trade_divergence_watch | -1 | -0.1067 | 0.6920 | -0.9055 | 10 | 1032 | 20816 | 0.1484 | 199392 | 12 | book imbalance and taker flow disagree |
| ADA | book_trade_divergence_watch | 1 | 0.0661 | -0.0794 | 0.2116 | 10 | 457 | 297 | 2.9126 | 93547 | 39 | book imbalance and taker flow disagree |
| NEAR | book_trade_divergence_watch | 1 | 0.0425 | 0.6736 | -0.5886 | 10 | 905 | 3494 | 0.4538 | 12782 | 20 | book imbalance and taker flow disagree |
| XMR | book_trade_divergence_watch | 1 | 0.0388 | 0.5315 | -0.4539 | 10 | 1337 | 3558 | 0.3168 | 7495 | 39 | book imbalance and taker flow disagree |
| STX | book_trade_divergence_watch | 1 | 0.0190 | 0.2463 | -0.2083 | 10 | 89 | 136 | 4.2592 | 2492 | 326 | book imbalance and taker flow disagree |
| STRK | wide_spread_watch | -1 | -0.1209 | 0.2929 | -0.5347 | 10 | 230 | 760 | 8.6096 | 4144 | 175 | spread is too wide for a first microstructure probe |

## Interpretation

`aligned_pressure_watch` means visible book pressure and recent taker flow point in the same direction. `book_trade_divergence_watch` means the book and taker flow disagree; that can be adverse selection or absorption, so it needs separate forward labels.
