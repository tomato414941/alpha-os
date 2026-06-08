# Current Microstructure Flow Snapshot

This joins Hyperliquid public book imbalance with recent trade-print imbalance. It is a short-horizon microstructure observation, not a deployable market-making model.

| asset | action | dir | pressure | book imb | trade imb | trades | buy USD | sell USD | spread bps | depth 10bps USD | window s | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| XMR | aligned_pressure_watch | 1 | 0.7535 | 0.5656 | 0.9413 | 10 | 2371 | 72 | 7.2469 | 5878 | 87 | book imbalance and taker flow point the same way |
| SEI | aligned_pressure_watch | 1 | 0.6242 | 0.2602 | 0.9883 | 10 | 2730 | 16 | 2.7779 | 8074 | 77 | book imbalance and taker flow point the same way |
| MON | aligned_pressure_watch | 1 | 0.5424 | 0.2154 | 0.8694 | 10 | 1070 | 75 | 1.3403 | 2729 | 115 | book imbalance and taker flow point the same way |
| HYPE | aligned_pressure_watch | 1 | 0.5001 | 0.0002 | 1.0000 | 10 | 16690 | 0 | 0.3081 | 168362 | 1 | book imbalance and taker flow point the same way |
| STRK | aligned_pressure_watch | 1 | 0.4468 | 0.5629 | 0.3307 | 10 | 906 | 456 | 5.6689 | 5158 | 85 | book imbalance and taker flow point the same way |
| ADA | book_trade_divergence_watch | 1 | 0.4690 | -0.0450 | 0.9830 | 10 | 15330 | 131 | 4.6808 | 39896 | 3 | book imbalance and taker flow disagree |
| BNB | book_trade_divergence_watch | 1 | 0.4433 | -0.0321 | 0.9188 | 10 | 1288 | 55 | 0.8256 | 80253 | 72 | book imbalance and taker flow disagree |
| SUI | book_trade_divergence_watch | 1 | 0.3888 | -0.2223 | 1.0000 | 10 | 6766 | 0 | 0.2591 | 106075 | 6 | book imbalance and taker flow disagree |
| POL | book_trade_divergence_watch | -1 | -0.2784 | 0.4433 | -1.0000 | 10 | 0 | 328 | 5.9245 | 2460 | 270 | book imbalance and taker flow disagree |
| BTC | book_trade_divergence_watch | 1 | 0.2760 | -0.4408 | 0.9927 | 10 | 51551 | 189 | 0.3116 | 6102437 | 1 | book imbalance and taker flow disagree |
| OP | book_trade_divergence_watch | 1 | 0.1963 | -0.6075 | 1.0000 | 10 | 2521 | 0 | 6.0686 | 1578 | 109 | book imbalance and taker flow disagree |
| NEAR | book_trade_divergence_watch | 1 | 0.1731 | -0.4341 | 0.7802 | 10 | 3453 | 426 | 4.5689 | 14945 | 19 | book imbalance and taker flow disagree |
| SOL | book_trade_divergence_watch | -1 | -0.1109 | -0.4114 | 0.1897 | 10 | 9945 | 6774 | 0.1476 | 345545 | 9 | book imbalance and taker flow disagree |
| APT | book_trade_divergence_watch | 1 | 0.0778 | 0.1689 | -0.0133 | 10 | 871 | 895 | 2.9338 | 10668 | 81 | book imbalance and taker flow disagree |
| ARB | book_trade_divergence_watch | -1 | -0.0063 | 0.5868 | -0.5994 | 10 | 159 | 633 | 2.3688 | 6663 | 124 | book imbalance and taker flow disagree |
| STX | no_clear_pressure | -1 | -0.0610 | -0.1385 | 0.0165 | 10 | 287 | 278 | 2.6201 | 3820 | 360 | book and taker-flow imbalance are both small |
| ETH | no_clear_pressure | 1 | 0.0193 | -0.0824 | 0.1210 | 10 | 19264 | 15106 | 0.5870 | 12735901 | 1 | book and taker-flow imbalance are both small |
| BERA | wide_spread_watch | 1 | 0.5550 | 0.2987 | 0.8112 | 10 | 822 | 86 | 13.8450 | 747 | 176 | spread is too wide for a first microstructure probe |
| CHIP | wide_spread_watch | 1 | 0.3122 | 0.0407 | 0.5837 | 10 | 687 | 180 | 10.3105 | 3913 | 77 | spread is too wide for a first microstructure probe |
| MEGA | wide_spread_watch | -1 | -0.0523 | -0.3200 | 0.2154 | 10 | 547 | 353 | 8.9963 | 1183 | 63 | spread is too wide for a first microstructure probe |

## Interpretation

`aligned_pressure_watch` means visible book pressure and recent taker flow point in the same direction. `book_trade_divergence_watch` means the book and taker flow disagree; that can be adverse selection or absorption, so it needs separate forward labels.
