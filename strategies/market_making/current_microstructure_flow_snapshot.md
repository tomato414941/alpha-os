# Current Microstructure Flow Snapshot

This joins Hyperliquid public book imbalance with recent trade-print imbalance. It is a short-horizon microstructure observation, not a deployable market-making model.

| asset | action | dir | pressure | book imb | trade imb | trades | buy USD | sell USD | spread bps | depth 10bps USD | window s | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SOL | aligned_pressure_watch | 1 | 0.7286 | 0.4572 | 1.0000 | 10 | 2029 | 0 | 0.1532 | 392639 | 6 | book imbalance and taker flow point the same way |
| XMR | aligned_pressure_watch | 1 | 0.5573 | 0.5397 | 0.5750 | 10 | 1837 | 496 | 1.6561 | 8113 | 52 | book imbalance and taker flow point the same way |
| MON | aligned_pressure_watch | 1 | 0.5502 | 0.5584 | 0.5421 | 10 | 283 | 84 | 4.6443 | 1655 | 172 | book imbalance and taker flow point the same way |
| BNB | aligned_pressure_watch | 1 | 0.5114 | 0.0821 | 0.9407 | 10 | 1010 | 31 | 0.3363 | 130447 | 65 | book imbalance and taker flow point the same way |
| NEAR | aligned_pressure_watch | 1 | 0.4815 | 0.4798 | 0.4831 | 10 | 1333 | 465 | 7.3055 | 12445 | 16 | book imbalance and taker flow point the same way |
| ARB | aligned_pressure_watch | 1 | 0.3874 | 0.0532 | 0.7216 | 10 | 1478 | 239 | 3.6903 | 21320 | 95 | book imbalance and taker flow point the same way |
| MEGA | aligned_pressure_watch | 1 | 0.3098 | 0.1551 | 0.4645 | 10 | 318 | 116 | 5.1650 | 4553 | 52 | book imbalance and taker flow point the same way |
| STX | aligned_pressure_watch | -1 | -0.2766 | -0.2751 | -0.2782 | 10 | 116 | 205 | 1.0837 | 2750 | 322 | book imbalance and taker flow point the same way |
| SEI | aligned_pressure_watch | 1 | 0.2139 | 0.1292 | 0.2986 | 10 | 105 | 57 | 2.2637 | 14742 | 240 | book imbalance and taker flow point the same way |
| SUI | aligned_pressure_watch | 1 | 0.1863 | 0.2828 | 0.0899 | 10 | 3516 | 2936 | 0.1362 | 75798 | 39 | book imbalance and taker flow point the same way |
| ETH | book_trade_divergence_watch | 1 | 0.4775 | -0.0449 | 1.0000 | 10 | 7010 | 0 | 0.6052 | 12888945 | 1 | book imbalance and taker flow disagree |
| POL | book_trade_divergence_watch | -1 | -0.4512 | 0.0976 | -1.0000 | 10 | 0 | 3545 | 4.0631 | 6306 | 70 | book imbalance and taker flow disagree |
| HYPE | book_trade_divergence_watch | 1 | 0.4358 | -0.1284 | 1.0000 | 10 | 20132 | 0 | 0.1629 | 186307 | 2 | book imbalance and taker flow disagree |
| APT | book_trade_divergence_watch | 1 | 0.4162 | -0.0996 | 0.9319 | 10 | 830 | 29 | 3.0391 | 15149 | 100 | book imbalance and taker flow disagree |
| ADA | book_trade_divergence_watch | -1 | -0.3325 | -0.6984 | 0.0334 | 10 | 607 | 568 | 0.6219 | 14505 | 49 | book imbalance and taker flow disagree |
| CHIP | book_trade_divergence_watch | 1 | 0.2801 | -0.2004 | 0.7606 | 10 | 449 | 61 | 5.6027 | 4059 | 433 | book imbalance and taker flow disagree |
| BTC | book_trade_divergence_watch | -1 | -0.2765 | 0.4196 | -0.9727 | 10 | 704 | 50786 | 0.1595 | 3332331 | 0 | book imbalance and taker flow disagree |
| BERA | book_trade_divergence_watch | 1 | 0.1182 | 0.3605 | -0.1241 | 10 | 91 | 117 | 5.0671 | 3751 | 398 | book imbalance and taker flow disagree |
| STRK | book_trade_divergence_watch | -1 | -0.0414 | -0.5768 | 0.4940 | 10 | 286 | 97 | 2.9891 | 3152 | 355 | book imbalance and taker flow disagree |
| OP | book_trade_divergence_watch | -1 | -0.0214 | 0.4189 | -0.4616 | 10 | 106 | 289 | 5.2513 | 3312 | 436 | book imbalance and taker flow disagree |

## Interpretation

`aligned_pressure_watch` means visible book pressure and recent taker flow point in the same direction. `book_trade_divergence_watch` means the book and taker flow disagree; that can be adverse selection or absorption, so it needs separate forward labels.
