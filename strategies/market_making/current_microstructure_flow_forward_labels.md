# Current Microstructure Flow Forward Labels

This labels book-plus-trade microstructure observations against Hyperliquid 15m and 1h forward returns. It is not net PnL.

| asset | action | dir | pressure | book imb | trade imb | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| OP | aligned_pressure_watch | -1 | -0.8124 | -0.6248 | -1.0000 |  |  |  |  | pending_15m |
| MON | aligned_pressure_watch | -1 | -0.5658 | -0.6503 | -0.4813 |  |  |  |  | pending_15m |
| HYPE | aligned_pressure_watch | 1 | 0.5514 | 0.3390 | 0.7637 |  |  |  |  | pending_15m |
| POL | aligned_pressure_watch | 1 | 0.5147 | 0.1705 | 0.8589 |  |  |  |  | pending_15m |
| ARB | book_trade_divergence_watch | -1 | -0.4645 | 0.0709 | -1.0000 |  |  |  |  | pending_15m |
| CHIP | aligned_pressure_watch | 1 | 0.4422 | 0.0133 | 0.8712 |  |  |  |  | pending_15m |
| BTC | aligned_pressure_watch | 1 | 0.4283 | 0.3480 | 0.5087 |  |  |  |  | pending_15m |
| ETH | book_trade_divergence_watch | -1 | -0.4235 | 0.1431 | -0.9900 |  |  |  |  | pending_15m |
| SUI | book_trade_divergence_watch | 1 | 0.4164 | -0.0967 | 0.9294 |  |  |  |  | pending_15m |
| BERA | aligned_pressure_watch | 1 | 0.3171 | 0.1517 | 0.4825 |  |  |  |  | pending_15m |
| BNB | aligned_pressure_watch | 1 | 0.2534 | 0.0700 | 0.4368 |  |  |  |  | pending_15m |
| MEGA | book_trade_divergence_watch | -1 | -0.2428 | 0.2373 | -0.7228 |  |  |  |  | pending_15m |
| SEI | book_trade_divergence_watch | 1 | 0.2013 | -0.2894 | 0.6920 |  |  |  |  | pending_15m |
| APT | book_trade_divergence_watch | 1 | 0.1326 | 0.6377 | -0.3725 |  |  |  |  | pending_15m |
| STRK | wide_spread_watch | -1 | -0.1209 | 0.2929 | -0.5347 |  |  |  |  | pending_15m |
| SOL | book_trade_divergence_watch | -1 | -0.1067 | 0.6920 | -0.9055 |  |  |  |  | pending_15m |
| ADA | book_trade_divergence_watch | 1 | 0.0661 | -0.0794 | 0.2116 |  |  |  |  | pending_15m |
| NEAR | book_trade_divergence_watch | 1 | 0.0425 | 0.6736 | -0.5886 |  |  |  |  | pending_15m |
| XMR | book_trade_divergence_watch | 1 | 0.0388 | 0.5315 | -0.4539 |  |  |  |  | pending_15m |
| STX | book_trade_divergence_watch | 1 | 0.0190 | 0.2463 | -0.2083 |  |  |  |  | pending_15m |

## Interpretation

Positive directional return means the microstructure direction was right before fees and slippage. Compare aligned pressure rows against book/trade divergence rows before promoting the feature.
