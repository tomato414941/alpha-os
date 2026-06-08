# Current Microstructure Flow Snapshot

This joins Hyperliquid public book imbalance with recent trade-print imbalance. It is a short-horizon microstructure observation, not a deployable market-making model.

| asset | action | dir | pressure | book imb | trade imb | trades | buy USD | sell USD | spread bps | depth 10bps USD | window s | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BTC | aligned_pressure_watch | 1 | 0.7027 | 0.4054 | 1.0000 | 10 | 36204 | 0 | 0.1564 | 3549315 | 2 | book imbalance and taker flow point the same way |
| NEAR | aligned_pressure_watch | -1 | -0.6307 | -0.4327 | -0.8287 | 10 | 306 | 3270 | 0.4588 | 14083 | 28 | book imbalance and taker flow point the same way |
| SOL | aligned_pressure_watch | -1 | -0.5829 | -0.2867 | -0.8792 | 10 | 3929 | 61109 | 0.1484 | 472311 | 4 | book imbalance and taker flow point the same way |
| XMR | aligned_pressure_watch | -1 | -0.5430 | -0.1217 | -0.9644 | 10 | 28 | 1556 | 5.3551 | 6807 | 16 | book imbalance and taker flow point the same way |
| POL | aligned_pressure_watch | 1 | 0.4933 | 0.2052 | 0.7813 | 10 | 2135 | 262 | 4.5755 | 4245 | 210 | book imbalance and taker flow point the same way |
| ADA | aligned_pressure_watch | -1 | -0.4082 | -0.0746 | -0.7418 | 10 | 330 | 2228 | 7.1098 | 56422 | 32 | book imbalance and taker flow point the same way |
| STX | aligned_pressure_watch | -1 | -0.3737 | -0.1819 | -0.5656 | 10 | 77 | 277 | 4.2119 | 3650 | 117 | book imbalance and taker flow point the same way |
| ARB | aligned_pressure_watch | 1 | 0.3364 | 0.5603 | 0.1124 | 10 | 1193 | 952 | 3.5695 | 7435 | 133 | book imbalance and taker flow point the same way |
| CHIP | aligned_pressure_watch | 1 | 0.3172 | 0.1502 | 0.4843 | 10 | 392 | 136 | 3.6931 | 4126 | 351 | book imbalance and taker flow point the same way |
| SEI | book_trade_divergence_watch | -1 | -0.4716 | 0.0492 | -0.9924 | 10 | 20 | 5241 | 3.3830 | 11752 | 22 | book imbalance and taker flow disagree |
| APT | book_trade_divergence_watch | -1 | -0.3905 | 0.1994 | -0.9803 | 10 | 11 | 1059 | 1.4726 | 9945 | 127 | book imbalance and taker flow disagree |
| STRK | book_trade_divergence_watch | 1 | 0.3780 | -0.0010 | 0.7569 | 10 | 482 | 67 | 5.6964 | 9785 | 143 | book imbalance and taker flow disagree |
| ETH | book_trade_divergence_watch | -1 | -0.3418 | 0.0707 | -0.7543 | 10 | 1240 | 8856 | 0.5899 | 11516676 | 1 | book imbalance and taker flow disagree |
| MON | book_trade_divergence_watch | -1 | -0.3210 | 0.2624 | -0.9043 | 10 | 92 | 1831 | 4.4831 | 3332 | 31 | book imbalance and taker flow disagree |
| BNB | book_trade_divergence_watch | -1 | -0.3154 | 0.1655 | -0.7963 | 10 | 152 | 1337 | 0.6624 | 107947 | 28 | book imbalance and taker flow disagree |
| HYPE | book_trade_divergence_watch | -1 | -0.2905 | 0.2176 | -0.7986 | 10 | 1852 | 16544 | 0.3099 | 111545 | 2 | book imbalance and taker flow disagree |
| SUI | book_trade_divergence_watch | -1 | -0.1076 | 0.2098 | -0.4250 | 10 | 1000 | 2479 | 1.4346 | 74493 | 44 | book imbalance and taker flow disagree |
| MEGA | book_trade_divergence_watch | 1 | 0.0628 | -0.1220 | 0.2476 | 10 | 432 | 260 | 3.5550 | 4542 | 120 | book imbalance and taker flow disagree |
| BERA | wide_spread_watch | 1 | 0.4643 | -0.0425 | 0.9711 | 10 | 2381 | 35 | 8.6869 | 2375 | 113 | spread is too wide for a first microstructure probe |
| OP | wide_spread_watch | -1 | -0.4490 | -0.8891 | -0.0089 | 10 | 394 | 401 | 8.1177 | 466 | 299 | spread is too wide for a first microstructure probe |

## Interpretation

`aligned_pressure_watch` means visible book pressure and recent taker flow point in the same direction. `book_trade_divergence_watch` means the book and taker flow disagree; that can be adverse selection or absorption, so it needs separate forward labels.
