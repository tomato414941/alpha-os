# Current OKX Liquidation Forward Labels

This labels liquidation-flow candidates with continuation returns. Positive continuation return means the forced-flow direction continued over that horizon.

| asset | action | dir | raw 15m | continuation 15m | raw 1h | continuation 1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ALLO | long_liquidation_cascade_watch | -1 | -0.015808 | 0.015808 |  |  |
| SOXL | long_liquidation_cascade_watch | -1 | -0.004261 | 0.004261 |  |  |
| BCH | short_liquidation_squeeze_watch | 1 | 0.002381 | 0.002381 |  |  |
| DOGE | long_liquidation_cascade_watch | -1 | -0.001611 | 0.001611 |  |  |
| CL | long_liquidation_cascade_watch | -1 | 0.000442 | -0.000442 | 0.001877 | -0.001877 |
| BTC | short_liquidation_squeeze_watch | 1 | -0.000727 | -0.000727 | -0.001127 | -0.001127 |
| SOL | short_liquidation_squeeze_watch | 1 | -0.001184 | -0.001184 |  |  |
| ETH | short_liquidation_squeeze_watch | 1 | -0.001973 | -0.001973 |  |  |
| MU | long_liquidation_cascade_watch | -1 | 0.002409 | -0.002409 |  |  |
| NEAR | long_liquidation_cascade_watch | -1 | 0.007309 | -0.007309 |  |  |
| CBRS | long_liquidation_cascade_watch | -1 | 0.007956 | -0.007956 |  |  |
| BSB | long_liquidation_cascade_watch | -1 | 0.009117 | -0.009117 |  |  |
| BEAT | long_liquidation_cascade_watch | -1 | 0.011999 | -0.011999 |  |  |
| H | long_liquidation_cascade_watch | -1 |  |  |  |  |
| PIPPIN | long_liquidation_cascade_watch | -1 |  |  |  |  |
| ZEC | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| MRVL | long_liquidation_cascade_watch | -1 |  |  |  |  |
| WLD | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| HYPE | mixed_liquidation_flow_watch | 0 | 0.000157 |  |  |  |

## Interpretation

This is price-only continuation labeling. It does not decide whether a liquidation event should be traded as continuation, reversal, or ignored without further regime and execution checks.
