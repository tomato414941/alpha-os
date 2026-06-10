# Current Microstructure Flow Snapshot

This joins Hyperliquid public book imbalance with recent trade-print imbalance. It is a short-horizon microstructure observation, not a deployable market-making model.

| asset | action | dir | pressure | book imb | trade imb | trades | buy USD | sell USD | spread bps | depth 10bps USD | window s | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BTC | aligned_pressure_watch | 1 | 0.9216 | 0.8432 | 1.0000 | 10 | 4746 | 0 | 0.1599 | 783015 | 0 | book imbalance and taker flow point the same way |
| ARB | aligned_pressure_watch | -1 | -0.6759 | -0.3517 | -1.0000 | 10 | 0 | 1351 | 1.2470 | 6371 | 72 | book imbalance and taker flow point the same way |
| ETH | aligned_pressure_watch | 1 | 0.6647 | 0.3413 | 0.9882 | 10 | 19638 | 117 | 0.6021 | 6342670 | 2 | book imbalance and taker flow point the same way |
| XMR | aligned_pressure_watch | 1 | 0.6301 | 0.2602 | 1.0000 | 10 | 1774 | 0 | 2.5625 | 14179 | 9 | book imbalance and taker flow point the same way |
| NEAR | aligned_pressure_watch | -1 | -0.5680 | -0.1361 | -1.0000 | 10 | 0 | 7027 | 4.3189 | 35463 | 13 | book imbalance and taker flow point the same way |
| MON | aligned_pressure_watch | 1 | 0.4831 | 0.5657 | 0.4005 | 10 | 1337 | 572 | 6.8584 | 5236 | 40 | book imbalance and taker flow point the same way |
| STRK | aligned_pressure_watch | -1 | -0.4157 | -0.2843 | -0.5471 | 10 | 63 | 214 | 6.0006 | 3228 | 326 | book imbalance and taker flow point the same way |
| SOL | book_trade_divergence_watch | -1 | -0.3940 | 0.0805 | -0.8685 | 10 | 510 | 7252 | 0.7652 | 982337 | 14 | book imbalance and taker flow disagree |
| OP | book_trade_divergence_watch | 1 | 0.3860 | 0.9444 | -0.1724 | 10 | 226 | 320 | 6.4233 | 700 | 268 | book imbalance and taker flow disagree |
| SUI | book_trade_divergence_watch | 1 | 0.3648 | -0.2703 | 1.0000 | 10 | 15051 | 0 | 1.3632 | 63881 | 6 | book imbalance and taker flow disagree |
| MEGA | book_trade_divergence_watch | 1 | 0.3125 | -0.1535 | 0.7785 | 10 | 613 | 76 | 4.1579 | 3061 | 254 | book imbalance and taker flow disagree |
| APT | book_trade_divergence_watch | 1 | 0.2993 | -0.2858 | 0.8844 | 10 | 3825 | 235 | 1.5417 | 7154 | 12 | book imbalance and taker flow disagree |
| BNB | book_trade_divergence_watch | 1 | 0.2663 | -0.1158 | 0.6484 | 10 | 1515 | 323 | 2.5203 | 109550 | 18 | book imbalance and taker flow disagree |
| HYPE | book_trade_divergence_watch | -1 | -0.2535 | 0.4015 | -0.9084 | 10 | 390 | 8118 | 0.3213 | 64559 | 2 | book imbalance and taker flow disagree |
| ADA | book_trade_divergence_watch | -1 | -0.0737 | 0.0046 | -0.1520 | 10 | 1113 | 1512 | 1.8002 | 71863 | 31 | book imbalance and taker flow disagree |
| SEI | book_trade_divergence_watch | 1 | 0.0618 | 0.4315 | -0.3079 | 10 | 479 | 905 | 2.4980 | 6479 | 150 | book imbalance and taker flow disagree |
| BERA | no_clear_pressure | -1 | -0.0879 | -0.0371 | -0.1387 | 10 | 713 | 943 | 2.4008 | 2873 | 129 | book and taker-flow imbalance are both small |
| STX | wide_spread_watch | -1 | -0.5050 | -0.4477 | -0.5623 | 10 | 175 | 623 | 8.8281 | 1582 | 139 | spread is too wide for a first microstructure probe |
| POL | wide_spread_watch | -1 | -0.4801 | -0.0734 | -0.8868 | 10 | 36 | 594 | 8.3682 | 3469 | 360 | spread is too wide for a first microstructure probe |
| CHIP | wide_spread_watch | 1 | 0.1747 | 0.2249 | 0.1246 | 10 | 172 | 134 | 8.2623 | 4490 | 331 | spread is too wide for a first microstructure probe |

## Interpretation

`aligned_pressure_watch` means visible book pressure and recent taker flow point in the same direction. `book_trade_divergence_watch` means the book and taker flow disagree; that can be adverse selection or absorption, so it needs separate forward labels.
