# Current Follow-Up Repeat Forward Labels

This labels source-specific follow-up observations against subsequent Hyperliquid returns. It is a repeat-observation label, not a PnL model.

| asset | source | action | dir | priority | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| WLD | okx_pressure | long_carry_discount_watch | 1 | 10.0571 |  |  |  |  | pending_15m |
| WLD | liquidation | short_liquidation_squeeze_watch | 1 | 10.0571 |  |  |  |  | pending_15m |
| ETH | okx_pressure | long_carry_discount_watch | 1 | 4.5510 |  |  |  |  | pending_15m |
| ETH | liquidation | short_liquidation_squeeze_watch | 1 | 4.5510 |  |  |  |  | pending_15m |
| ETH | l2_imbalance | visible_book_imbalance | 1 | 4.5510 |  |  |  |  | pending_15m |
| BTC | liquidation | short_liquidation_squeeze_watch | 1 | 3.6217 |  |  |  |  | pending_15m |
| BTC | l2_imbalance | visible_book_imbalance | 1 | 3.6217 |  |  |  |  | pending_15m |
| ONDO | liquidation | short_liquidation_squeeze_watch | 1 | 3.6106 |  |  |  |  | pending_15m |
| ONDO | sector_rotation | sector_momentum_watch | 1 | 3.6106 |  |  |  |  | pending_15m |
| XRP | okx_pressure | long_carry_discount_watch | 1 | 3.4627 |  |  |  |  | pending_15m |
| XRP | liquidation | short_liquidation_squeeze_watch | 1 | 3.4627 |  |  |  |  | pending_15m |
| XPL | l2_imbalance | visible_book_imbalance | 1 | 3.4493 |  |  |  |  | pending_15m |
| XPL | sector_rotation | sector_momentum_watch | 1 | 3.4493 |  |  |  |  | pending_15m |
| LTC | okx_pressure | long_carry_discount_watch | 1 | 3.2959 |  |  |  |  | pending_15m |
| LTC | liquidation | long_liquidation_cascade_watch | -1 | 3.2959 |  |  |  |  | pending_15m |
| SOL | okx_pressure | long_carry_discount_watch | 1 | 3.1187 |  |  |  |  | pending_15m |
| SOL | liquidation | short_liquidation_squeeze_watch | 1 | 3.1187 |  |  |  |  | pending_15m |
| PUMP | liquidation | short_liquidation_squeeze_watch | 1 | 2.9792 |  |  |  |  | pending_15m |
| PUMP | sector_rotation | sector_momentum_watch | 1 | 2.9792 |  |  |  |  | pending_15m |
| XLM | okx_pressure | long_carry_discount_watch | 1 | 2.9178 |  |  |  |  | pending_15m |
| XLM | l2_imbalance | visible_book_imbalance | 1 | 2.9178 |  |  |  |  | pending_15m |
| TON | okx_pressure | short_carry_watch | -1 | 2.1872 |  |  |  |  | pending_15m |
| TON | liquidation | short_liquidation_squeeze_watch | 1 | 2.1872 |  |  |  |  | pending_15m |

## Interpretation

`pending_15m` or `pending_1h` means the observation has not matured. Positive directional return means the source-specific direction was right over that horizon before fees, funding PnL, and slippage.
