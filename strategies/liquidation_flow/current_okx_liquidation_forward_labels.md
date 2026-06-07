# Current OKX Liquidation Forward Labels

This labels liquidation-flow candidates with continuation returns. Positive continuation return means the forced-flow direction continued over that horizon.

| asset | action | dir | raw 15m | continuation 15m | raw 1h | continuation 1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HOME | short_liquidation_squeeze_watch | 1 | 0.020731 | 0.020731 |  |  |
| TON | short_liquidation_squeeze_watch | 1 | 0.002904 | 0.002904 |  |  |
| H | long_liquidation_cascade_watch | -1 | -0.002635 | 0.002635 |  |  |
| BTC | short_liquidation_squeeze_watch | 1 | 0.000312 | 0.000312 |  |  |
| MRVL | long_liquidation_cascade_watch | -1 | 0.000354 | -0.000354 |  |  |
| SUI | short_liquidation_squeeze_watch | 1 | -0.000400 | -0.000400 |  |  |
| ADA | short_liquidation_squeeze_watch | 1 | -0.000612 | -0.000612 |  |  |
| ETH | short_liquidation_squeeze_watch | 1 | -0.001378 | -0.001378 |  |  |
| FIL | long_liquidation_cascade_watch | -1 | 0.002504 | -0.002504 |  |  |
| XLM | long_liquidation_cascade_watch | -1 | 0.003891 | -0.003891 |  |  |
| NEAR | long_liquidation_cascade_watch | -1 | 0.004420 | -0.004420 |  |  |
| ONDO | long_liquidation_cascade_watch | -1 | 0.008452 | -0.008452 |  |  |
| ALLO | long_liquidation_cascade_watch | -1 | 0.011002 | -0.011002 |  |  |
| BEAT | long_liquidation_cascade_watch | -1 | 0.014449 | -0.014449 |  |  |
| OPN | long_liquidation_cascade_watch | -1 |  |  |  |  |
| XRP | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| ZEC | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| JTO | mixed_liquidation_flow_watch | 0 | -0.001786 |  |  |  |
| WLD | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| BSB | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| LAB | long_liquidation_cascade_watch | -1 |  |  |  |  |
| HYPE | mixed_liquidation_flow_watch | 0 |  |  |  |  |

## Interpretation

This is price-only continuation labeling. It does not decide whether a liquidation event should be traded as continuation, reversal, or ignored without further regime and execution checks.
