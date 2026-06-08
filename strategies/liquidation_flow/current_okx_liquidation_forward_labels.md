# Current OKX Liquidation Forward Labels

This labels liquidation-flow candidates with continuation returns. Positive continuation return means the forced-flow direction continued over that horizon.

| asset | action | dir | raw 15m | continuation 15m | raw 1h | continuation 1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TON | short_liquidation_squeeze_watch | 1 | 0.004640 | 0.004640 |  |  |
| NEAR | short_liquidation_squeeze_watch | 1 | 0.004153 | 0.004153 |  |  |
| MU | long_liquidation_cascade_watch | -1 | -0.002697 | 0.002697 |  |  |
| CL | short_liquidation_squeeze_watch | 1 | 0.001207 | 0.001207 |  |  |
| XRP | short_liquidation_squeeze_watch | 1 | 0.000854 | 0.000854 |  |  |
| ONDO | short_liquidation_squeeze_watch | 1 | 0.000544 | 0.000544 |  |  |
| XAU | long_liquidation_cascade_watch | -1 | 0.000555 | -0.000555 | 0.003494 | -0.003494 |
| MRVL | short_liquidation_squeeze_watch | 1 | -0.001005 | -0.001005 |  |  |
| HYPE | long_liquidation_cascade_watch | -1 | 0.001238 | -0.001238 |  |  |
| SNDK | short_liquidation_squeeze_watch | 1 | -0.002385 | -0.002385 |  |  |
| FIL | long_liquidation_cascade_watch | -1 | 0.002700 | -0.002700 |  |  |
| XAG | long_liquidation_cascade_watch | -1 | 0.002798 | -0.002798 | 0.007806 | -0.007806 |
| HOME | long_liquidation_cascade_watch | -1 | 0.009782 | -0.009782 |  |  |
| BCH | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| ETH | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| BSB | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| WLD | long_liquidation_cascade_watch | -1 |  |  |  |  |
| BTC | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| ZEC | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| BNB | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| DOGE | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| ALLO | mixed_liquidation_flow_watch | 0 | 0.006709 |  |  |  |
| SOL | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| ADA | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| BEAT | mixed_liquidation_flow_watch | 0 | -0.001483 |  |  |  |

## Interpretation

This is price-only continuation labeling. It does not decide whether a liquidation event should be traded as continuation, reversal, or ignored without further regime and execution checks.
