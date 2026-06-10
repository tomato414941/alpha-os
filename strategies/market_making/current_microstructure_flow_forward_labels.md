# Current Microstructure Flow Forward Labels

This labels book-plus-trade microstructure observations against Hyperliquid 15m and 1h forward returns. It is not net PnL.

| asset | action | dir | pressure | book imb | trade imb | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BTC | aligned_pressure_watch | 1 | 0.9216 | 0.8432 | 1.0000 |  |  |  |  | pending_15m |
| ARB | aligned_pressure_watch | -1 | -0.6759 | -0.3517 | -1.0000 |  |  |  |  | pending_15m |
| ETH | aligned_pressure_watch | 1 | 0.6647 | 0.3413 | 0.9882 |  |  |  |  | pending_15m |
| XMR | aligned_pressure_watch | 1 | 0.6301 | 0.2602 | 1.0000 |  |  |  |  | pending_15m |
| NEAR | aligned_pressure_watch | -1 | -0.5680 | -0.1361 | -1.0000 |  |  |  |  | pending_15m |
| STX | wide_spread_watch | -1 | -0.5050 | -0.4477 | -0.5623 |  |  |  |  | pending_15m |
| MON | aligned_pressure_watch | 1 | 0.4831 | 0.5657 | 0.4005 |  |  |  |  | pending_15m |
| POL | wide_spread_watch | -1 | -0.4801 | -0.0734 | -0.8868 |  |  |  |  | pending_15m |
| STRK | aligned_pressure_watch | -1 | -0.4157 | -0.2843 | -0.5471 |  |  |  |  | pending_15m |
| SOL | book_trade_divergence_watch | -1 | -0.3940 | 0.0805 | -0.8685 |  |  |  |  | pending_15m |
| OP | book_trade_divergence_watch | 1 | 0.3860 | 0.9444 | -0.1724 |  |  |  |  | pending_15m |
| SUI | book_trade_divergence_watch | 1 | 0.3648 | -0.2703 | 1.0000 |  |  |  |  | pending_15m |
| MEGA | book_trade_divergence_watch | 1 | 0.3125 | -0.1535 | 0.7785 |  |  |  |  | pending_15m |
| APT | book_trade_divergence_watch | 1 | 0.2993 | -0.2858 | 0.8844 |  |  |  |  | pending_15m |
| BNB | book_trade_divergence_watch | 1 | 0.2663 | -0.1158 | 0.6484 |  |  |  |  | pending_15m |
| HYPE | book_trade_divergence_watch | -1 | -0.2535 | 0.4015 | -0.9084 |  |  |  |  | pending_15m |
| CHIP | wide_spread_watch | 1 | 0.1747 | 0.2249 | 0.1246 |  |  |  |  | pending_15m |
| BERA | no_clear_pressure | -1 | -0.0879 | -0.0371 | -0.1387 |  |  |  |  | pending_15m |
| ADA | book_trade_divergence_watch | -1 | -0.0737 | 0.0046 | -0.1520 |  |  |  |  | pending_15m |
| SEI | book_trade_divergence_watch | 1 | 0.0618 | 0.4315 | -0.3079 |  |  |  |  | pending_15m |

## Interpretation

Positive directional return means the microstructure direction was right before fees and slippage. Compare aligned pressure rows against book/trade divergence rows before promoting the feature.
