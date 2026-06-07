# Current OKX Liquidation Forward Labels

This labels liquidation-flow candidates with continuation returns. Positive continuation return means the forced-flow direction continued over that horizon.

| asset | action | dir | raw 15m | continuation 15m | raw 1h | continuation 1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BEAT | long_liquidation_cascade_watch | -1 | -0.023958 | 0.023958 |  |  |
| OPN | long_liquidation_cascade_watch | -1 | -0.004695 | 0.004695 |  |  |
| H | long_liquidation_cascade_watch | -1 | -0.002635 | 0.002635 | 0.030853 | -0.030853 |
| JTO | long_liquidation_cascade_watch | -1 | -0.002276 | 0.002276 |  |  |
| BTC | short_liquidation_squeeze_watch | 1 | 0.000312 | 0.000312 | -0.000653 | -0.000653 |
| SUI | short_liquidation_squeeze_watch | 1 | -0.000400 | -0.000400 |  |  |
| ADA | short_liquidation_squeeze_watch | 1 | -0.000612 | -0.000612 | -0.007339 | -0.007339 |
| MRVL | long_liquidation_cascade_watch | -1 | 0.000708 | -0.000708 |  |  |
| ETH | short_liquidation_squeeze_watch | 1 | -0.001378 | -0.001378 |  |  |
| FIL | long_liquidation_cascade_watch | -1 | 0.002636 | -0.002636 |  |  |
| XLM | long_liquidation_cascade_watch | -1 | 0.003891 | -0.003891 | 0.006323 | -0.006323 |
| XRP | short_liquidation_squeeze_watch | 1 | -0.004348 | -0.004348 |  |  |
| NEAR | long_liquidation_cascade_watch | -1 | 0.004912 | -0.004912 |  |  |
| ONDO | long_liquidation_cascade_watch | -1 | 0.009327 | -0.009327 |  |  |
| ALLO | long_liquidation_cascade_watch | -1 | 0.011002 | -0.011002 |  |  |
| TON | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| BSB | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| ZEC | mixed_liquidation_flow_watch | 0 | -0.002418 |  |  |  |
| HOME | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| WLD | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| LAB | long_liquidation_cascade_watch | -1 |  |  |  |  |
| HYPE | mixed_liquidation_flow_watch | 0 | -0.000842 |  |  |  |

## Interpretation

This is price-only continuation labeling. It does not decide whether a liquidation event should be traded as continuation, reversal, or ignored without further regime and execution checks.
