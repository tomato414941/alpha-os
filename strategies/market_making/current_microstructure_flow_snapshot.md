# Current Microstructure Flow Snapshot

This joins Hyperliquid public book imbalance with recent trade-print imbalance. It is a short-horizon microstructure observation, not a deployable market-making model.

| asset | action | dir | pressure | book imb | trade imb | trades | buy USD | sell USD | spread bps | depth 10bps USD | window s | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEAR | aligned_pressure_watch | 1 | 0.8025 | 0.6523 | 0.9526 | 10 | 21588 | 524 | 0.4643 | 16700 | 6 | book imbalance and taker flow point the same way |
| HYPE | aligned_pressure_watch | 1 | 0.7656 | 0.5313 | 1.0000 | 10 | 39619 | 0 | 0.1614 | 119267 | 0 | book imbalance and taker flow point the same way |
| SOL | aligned_pressure_watch | 1 | 0.7038 | 0.4076 | 1.0000 | 10 | 22031 | 0 | 0.6012 | 563534 | 7 | book imbalance and taker flow point the same way |
| SEI | aligned_pressure_watch | 1 | 0.7019 | 0.4038 | 1.0000 | 10 | 3471 | 0 | 4.5932 | 6862 | 2 | book imbalance and taker flow point the same way |
| BTC | aligned_pressure_watch | 1 | 0.5821 | 0.1643 | 1.0000 | 10 | 69460 | 1 | 0.1582 | 6431380 | 0 | book imbalance and taker flow point the same way |
| ETH | aligned_pressure_watch | 1 | 0.5055 | 0.1090 | 0.9020 | 10 | 10547 | 544 | 0.5952 | 11207347 | 6 | book imbalance and taker flow point the same way |
| ADA | aligned_pressure_watch | 1 | 0.4817 | 0.0589 | 0.9045 | 10 | 1291 | 65 | 1.2019 | 70371 | 45 | book imbalance and taker flow point the same way |
| ARB | aligned_pressure_watch | -1 | -0.4561 | -0.4803 | -0.4319 | 10 | 408 | 1029 | 2.4030 | 8932 | 111 | book imbalance and taker flow point the same way |
| POL | aligned_pressure_watch | 1 | 0.4213 | 0.2801 | 0.5625 | 10 | 950 | 266 | 5.4280 | 5077 | 563 | book imbalance and taker flow point the same way |
| MEGA | aligned_pressure_watch | -1 | -0.3196 | -0.2932 | -0.3459 | 10 | 239 | 493 | 2.1937 | 4191 | 30 | book imbalance and taker flow point the same way |
| OP | aligned_pressure_watch | -1 | -0.3140 | -0.0466 | -0.5814 | 10 | 130 | 493 | 7.0839 | 1821 | 322 | book imbalance and taker flow point the same way |
| BERA | aligned_pressure_watch | 1 | 0.2569 | 0.3540 | 0.1598 | 10 | 493 | 357 | 7.0489 | 820 | 57 | book imbalance and taker flow point the same way |
| STX | aligned_pressure_watch | 1 | 0.2287 | 0.2079 | 0.2495 | 10 | 227 | 136 | 4.7658 | 3655 | 372 | book imbalance and taker flow point the same way |
| MON | book_trade_divergence_watch | 1 | 0.3612 | 0.7653 | -0.0429 | 10 | 284 | 309 | 2.2757 | 3597 | 120 | book imbalance and taker flow disagree |
| STRK | book_trade_divergence_watch | -1 | -0.2869 | 0.4048 | -0.9787 | 10 | 15 | 1402 | 5.7921 | 5455 | 142 | book imbalance and taker flow disagree |
| CHIP | book_trade_divergence_watch | 1 | 0.2583 | 0.5580 | -0.0413 | 10 | 110 | 119 | 7.5115 | 2386 | 366 | book imbalance and taker flow disagree |
| XMR | book_trade_divergence_watch | -1 | -0.2398 | 0.0935 | -0.5731 | 10 | 153 | 565 | 1.9126 | 8379 | 31 | book imbalance and taker flow disagree |
| SUI | book_trade_divergence_watch | -1 | -0.1921 | 0.3782 | -0.7623 | 10 | 233 | 1725 | 0.1318 | 79609 | 55 | book imbalance and taker flow disagree |
| BNB | book_trade_divergence_watch | 1 | 0.0991 | 0.2368 | -0.0385 | 10 | 628 | 678 | 1.4997 | 99878 | 22 | book imbalance and taker flow disagree |
| APT | book_trade_divergence_watch | 1 | 0.0110 | 0.3716 | -0.3495 | 10 | 946 | 1962 | 1.4742 | 15390 | 99 | book imbalance and taker flow disagree |

## Interpretation

`aligned_pressure_watch` means visible book pressure and recent taker flow point in the same direction. `book_trade_divergence_watch` means the book and taker flow disagree; that can be adverse selection or absorption, so it needs separate forward labels.
