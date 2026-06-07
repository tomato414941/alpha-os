# Current HL Signal Forward Labels

This labels elapsed monitor samples with subsequent Hyperliquid candle returns. It is a small forward-label check, not a final alpha test.

| source | action | asset | obs | cov15 | cov1h | mean 15m | mean 1h | hit15 | hit1h |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| okx_hl_current | paper_24h_monitor | STABLE | 12 | 12 | 12 | 0.000000 | 0.002111 | 0.000000 | 1.000000 |
| okx_hl_current | paper_24h_monitor | WLD | 10 | 10 | 10 | 0.000000 | 0.051314 | 0.000000 | 1.000000 |
| perp_carry_reversion | long_carry_reversion_watch | SNX | 6 | 6 | 6 | 0.000869 | -0.018447 | 1.000000 | 0.000000 |
| perp_carry_reversion | short_carry_reversion_watch | AZTEC | 6 | 6 | 6 | 0.000758 | -0.024080 | 1.000000 | 0.000000 |
| perp_carry_reversion | short_carry_reversion_watch | APEX | 6 | 6 | 6 | -0.000494 | -0.030341 | 0.000000 | 0.000000 |
| perp_carry_reversion | long_carry_reversion_watch | ETC | 6 | 6 | 6 | -0.001297 | -0.019105 | 0.000000 | 0.000000 |
| perp_carry_reversion | long_carry_reversion_watch | ATOM | 6 | 6 | 6 | -0.002017 | -0.020528 | 0.000000 | 0.000000 |
| perp_carry_reversion | long_carry_reversion_watch | MET | 6 | 6 | 6 | -0.002864 | -0.029846 | 0.000000 | 0.000000 |
| perp_carry_reversion | long_carry_reversion_watch | BSV | 6 | 6 | 6 | -0.003004 | -0.015689 | 0.000000 | 0.000000 |
| perp_carry_reversion | long_carry_reversion_watch | UMA | 6 | 6 | 6 | -0.003458 | -0.023002 | 0.000000 | 0.000000 |
| perp_carry_reversion | short_carry_reversion_watch | ZRO | 6 | 6 | 6 | -0.003695 | -0.028771 | 0.000000 | 0.000000 |
| perp_carry_reversion | short_carry_reversion_watch | CFX | 6 | 6 | 6 | -0.003753 | -0.025314 | 0.000000 | 0.000000 |
| perp_carry_reversion | short_carry_reversion_watch | PURR | 6 | 6 | 6 | -0.003900 | -0.019158 | 0.000000 | 0.000000 |
| perp_carry_reversion | short_carry_reversion_watch | DYDX | 6 | 6 | 6 | -0.005056 | -0.041573 | 0.000000 | 0.000000 |
| perp_carry_reversion | long_carry_reversion_watch | NIL | 6 | 6 | 6 | -0.005189 | -0.039423 | 0.000000 | 0.000000 |
| perp_carry_reversion | short_carry_reversion_watch | HEMI | 6 | 6 | 6 | -0.005240 | -0.025117 | 0.000000 | 0.000000 |
| perp_carry_reversion | short_carry_reversion_watch | HMSTR | 6 | 6 | 6 | -0.005405 | -0.005405 | 0.000000 | 0.000000 |
| perp_carry_reversion | long_carry_reversion_watch | TRUMP | 6 | 6 | 6 | -0.005622 | -0.027257 | 0.000000 | 0.000000 |
| perp_carry_reversion | long_carry_reversion_watch | AAVE | 6 | 6 | 6 | -0.005690 | -0.022503 | 0.000000 | 0.000000 |
| perp_carry_reversion | short_carry_reversion_watch | STBL | 6 | 6 | 6 | -0.005820 | 0.000000 | 0.000000 | 0.000000 |
| perp_carry_reversion | long_carry_reversion_watch | SEI | 6 | 6 | 6 | -0.006884 | -0.028513 | 0.000000 | 0.000000 |
| perp_carry_reversion | long_carry_reversion_watch | VIRTUAL | 6 | 6 | 6 | -0.007280 | -0.030414 | 0.000000 | 0.000000 |
| perp_carry_reversion | short_carry_reversion_watch | XMR | 6 | 6 | 6 | -0.007301 | -0.025261 | 0.000000 | 0.000000 |
| perp_carry_reversion | short_carry_reversion_watch | GRIFFAIN | 6 | 6 | 6 | -0.007559 | -0.027633 | 0.000000 | 0.000000 |
| perp_carry_reversion | long_carry_reversion_watch | POPCAT | 6 | 6 | 6 | -0.012904 | -0.034158 | 0.000000 | 0.000000 |

## Interpretation

This labels price movement after signal timestamps only. It does not yet include funding PnL, hedge PnL, fees, adverse selection, or neutral baselines.
