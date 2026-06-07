# Current HL Signal Forward Labels

This labels elapsed monitor samples with subsequent Hyperliquid candle returns. It is a small forward-label check, not a final alpha test.

| source | action | asset | obs | cov15 | cov1h | mean 15m | mean 1h | hit15 | hit1h |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| okx_hl_current | paper_24h_monitor | STABLE | 12 | 12 | 0 | -0.001692 |  | 0.000000 |  |
| okx_hl_current | paper_24h_monitor | WLD | 10 | 10 | 0 | 0.019682 |  | 1.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | MEGA | 6 | 6 | 0 | 0.017831 |  | 1.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | IP | 6 | 6 | 0 | 0.015990 |  | 1.000000 |  |
| perp_carry_reversion | short_carry_reversion_watch | XMR | 6 | 6 | 0 | 0.011059 |  | 1.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | ZORA | 6 | 6 | 0 | 0.005486 |  | 1.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | BSV | 6 | 6 | 0 | 0.001747 |  | 1.000000 |  |
| perp_carry_reversion | short_carry_reversion_watch | ZRO | 6 | 6 | 0 | 0.001267 |  | 1.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | SAGA | 6 | 6 | 0 | 0.000729 |  | 1.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | JUP | 6 | 6 | 0 | -0.000385 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | SNX | 6 | 6 | 0 | -0.000579 |  | 0.000000 |  |
| perp_carry_reversion | short_carry_reversion_watch | PURR | 6 | 6 | 0 | -0.000747 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | BABY | 6 | 6 | 0 | -0.001093 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | POPCAT | 6 | 6 | 0 | -0.001370 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | TRUMP | 6 | 6 | 0 | -0.001716 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | STABLE | 6 | 6 | 0 | -0.001784 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | AERO | 6 | 6 | 0 | -0.001786 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | ATOM | 6 | 6 | 0 | -0.002192 |  | 0.000000 |  |
| perp_carry_reversion | short_carry_reversion_watch | HEMI | 6 | 6 | 0 | -0.002858 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | MON | 6 | 6 | 0 | -0.002903 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | SEI | 6 | 6 | 0 | -0.004280 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | MORPHO | 6 | 6 | 0 | -0.004331 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | BIO | 6 | 6 | 0 | -0.007837 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | OP | 5 | 5 | 0 | -0.000104 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | XAI | 5 | 5 | 0 | -0.003797 |  | 0.000000 |  |

## Interpretation

This labels price movement after signal timestamps only. It does not yet include funding PnL, hedge PnL, fees, adverse selection, or neutral baselines.
