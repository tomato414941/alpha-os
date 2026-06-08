# Current Microstructure Flow Forward Labels

This labels book-plus-trade microstructure observations against Hyperliquid 15m and 1h forward returns. It is not net PnL.

| asset | action | dir | pressure | book imb | trade imb | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| XMR | aligned_pressure_watch | 1 | 0.7535 | 0.5656 | 0.9413 |  |  |  |  | pending_15m |
| SEI | aligned_pressure_watch | 1 | 0.6242 | 0.2602 | 0.9883 |  |  |  |  | pending_15m |
| BERA | wide_spread_watch | 1 | 0.5550 | 0.2987 | 0.8112 |  |  |  |  | pending_15m |
| MON | aligned_pressure_watch | 1 | 0.5424 | 0.2154 | 0.8694 |  |  |  |  | pending_15m |
| HYPE | aligned_pressure_watch | 1 | 0.5001 | 0.0002 | 1.0000 |  |  |  |  | pending_15m |
| ADA | book_trade_divergence_watch | 1 | 0.4690 | -0.0450 | 0.9830 |  |  |  |  | pending_15m |
| STRK | aligned_pressure_watch | 1 | 0.4468 | 0.5629 | 0.3307 |  |  |  |  | pending_15m |
| BNB | book_trade_divergence_watch | 1 | 0.4433 | -0.0321 | 0.9188 |  |  |  |  | pending_15m |
| SUI | book_trade_divergence_watch | 1 | 0.3888 | -0.2223 | 1.0000 |  |  |  |  | pending_15m |
| CHIP | wide_spread_watch | 1 | 0.3122 | 0.0407 | 0.5837 |  |  |  |  | pending_15m |
| POL | book_trade_divergence_watch | -1 | -0.2784 | 0.4433 | -1.0000 |  |  |  |  | pending_15m |
| BTC | book_trade_divergence_watch | 1 | 0.2760 | -0.4408 | 0.9927 |  |  |  |  | pending_15m |
| OP | book_trade_divergence_watch | 1 | 0.1963 | -0.6075 | 1.0000 |  |  |  |  | pending_15m |
| NEAR | book_trade_divergence_watch | 1 | 0.1731 | -0.4341 | 0.7802 |  |  |  |  | pending_15m |
| SOL | book_trade_divergence_watch | -1 | -0.1109 | -0.4114 | 0.1897 |  |  |  |  | pending_15m |
| APT | book_trade_divergence_watch | 1 | 0.0778 | 0.1689 | -0.0133 |  |  |  |  | pending_15m |
| STX | no_clear_pressure | -1 | -0.0610 | -0.1385 | 0.0165 |  |  |  |  | pending_15m |
| MEGA | wide_spread_watch | -1 | -0.0523 | -0.3200 | 0.2154 |  |  |  |  | pending_15m |
| ETH | no_clear_pressure | 1 | 0.0193 | -0.0824 | 0.1210 |  |  |  |  | pending_15m |
| ARB | book_trade_divergence_watch | -1 | -0.0063 | 0.5868 | -0.5994 |  |  |  |  | pending_15m |

## Interpretation

Positive directional return means the microstructure direction was right before fees and slippage. Compare aligned pressure rows against book/trade divergence rows before promoting the feature.
