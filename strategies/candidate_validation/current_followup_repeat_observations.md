# Current Follow-Up Repeat Observations

This records fresh source-specific observations from the follow-up queue. Each row is asset x source, so mixed evidence is not averaged together before labeling.

| asset | source | source action | dir | priority | mark | funding ann | spread bps | depth 10bps USD | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| WLD | hl_candidate |  | 0 | 10.0571 | 0.48648000 | -0.214318 | 5.3436 | 22766 | missing_source_direction |
| WLD | okx_pressure | long_carry_discount_watch | 1 | 10.0571 | 0.48648000 | -0.214318 | 5.3436 | 22766 | ready_for_label |
| WLD | liquidation | short_liquidation_squeeze_watch | 1 | 10.0571 | 0.48648000 | -0.214318 | 5.3436 | 22766 | ready_for_label |
| ETH | okx_pressure | long_carry_discount_watch | 1 | 4.5510 | 1633.00000000 | 0.065543 | 0.6123 | 12184113 | ready_for_label |
| ETH | liquidation | short_liquidation_squeeze_watch | 1 | 4.5510 | 1633.00000000 | 0.065543 | 0.6123 | 12184113 | ready_for_label |
| ETH | l2_imbalance | visible_book_imbalance | 1 | 4.5510 | 1633.00000000 | 0.065543 | 0.6123 | 12184113 | ready_for_label |
| BTC | liquidation | short_liquidation_squeeze_watch | 1 | 3.6217 | 62244.00000000 | 0.109500 | 0.1607 | 2835549 | ready_for_label |
| BTC | l2_imbalance | visible_book_imbalance | 1 | 3.6217 | 62244.00000000 | 0.109500 | 0.1607 | 2835549 | ready_for_label |
| ONDO | liquidation | short_liquidation_squeeze_watch | 1 | 3.6106 | 0.34858000 | 0.109500 | 0.8600 | 35880 | ready_for_label |
| ONDO | sector_rotation | sector_momentum_watch | 1 | 3.6106 | 0.34858000 | 0.109500 | 0.8600 | 35880 | ready_for_label |
| XRP | okx_pressure | long_carry_discount_watch | 1 | 3.4627 | 1.14390000 | 0.026255 | 0.8741 | 525602 | ready_for_label |
| XRP | liquidation | short_liquidation_squeeze_watch | 1 | 3.4627 | 1.14390000 | 0.026255 | 0.8741 | 525602 | ready_for_label |
| XPL | l2_imbalance | visible_book_imbalance | 1 | 3.4493 | 0.06900700 | 0.109500 | 3.3322 | 5094 | ready_for_label |
| XPL | sector_rotation | sector_momentum_watch | 1 | 3.4493 | 0.06900700 | 0.109500 | 3.3322 | 5094 | ready_for_label |
| LTC | okx_pressure | long_carry_discount_watch | 1 | 3.2959 | 42.01600000 | 0.109500 | 0.9519 | 55182 | ready_for_label |
| LTC | liquidation | long_liquidation_cascade_watch | -1 | 3.2959 | 42.01600000 | 0.109500 | 0.9519 | 55182 | ready_for_label |
| SOL | okx_pressure | long_carry_discount_watch | 1 | 3.1187 | 65.40200000 | -0.193780 | 0.1529 | 411839 | ready_for_label |
| SOL | liquidation | short_liquidation_squeeze_watch | 1 | 3.1187 | 65.40200000 | -0.193780 | 0.1529 | 411839 | ready_for_label |
| PUMP | liquidation | short_liquidation_squeeze_watch | 1 | 2.9792 | 0.00151000 | 0.109500 | 6.6203 | 37064 | ready_for_label |
| PUMP | sector_rotation | sector_momentum_watch | 1 | 2.9792 | 0.00151000 | 0.109500 | 6.6203 | 37064 | ready_for_label |
| XLM | okx_pressure | long_carry_discount_watch | 1 | 2.9178 | 0.20878000 | 0.109500 | 6.2268 | 8613 | ready_for_label |
| XLM | l2_imbalance | visible_book_imbalance | 1 | 2.9178 | 0.20878000 | 0.109500 | 6.2268 | 8613 | ready_for_label |
| TON | okx_pressure | short_carry_watch | -1 | 2.1872 | 1.71670000 | 0.109500 | 2.3301 | 16181 | ready_for_label |
| TON | liquidation | short_liquidation_squeeze_watch | 1 | 2.1872 | 1.71670000 | 0.109500 | 2.3301 | 16181 | ready_for_label |

## Interpretation

`ready_for_label` means the source had a direction and can be labeled after 15m/1h. `missing_source_direction` keeps the context visible but does not create a directional alpha label.
