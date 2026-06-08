# Current OKX Liquidation Forward Labels

This labels liquidation-flow candidates with continuation returns. Positive continuation return means the forced-flow direction continued over that horizon.

| asset | action | dir | raw 15m | continuation 15m | raw 1h | continuation 1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NEAR | short_liquidation_squeeze_watch | 1 | 0.009229 | 0.009229 |  |  |
| ONDO | short_liquidation_squeeze_watch | 1 | 0.002177 | 0.002177 |  |  |
| BEAT | short_liquidation_squeeze_watch | 1 | 0.001620 | 0.001620 |  |  |
| MRVL | short_liquidation_squeeze_watch | 1 | 0.001340 | 0.001340 |  |  |
| CL | short_liquidation_squeeze_watch | 1 | 0.001207 | 0.001207 |  |  |
| HYPE | long_liquidation_cascade_watch | -1 | 0.001238 | -0.001238 |  |  |
| FIL | long_liquidation_cascade_watch | -1 | 0.002700 | -0.002700 |  |  |
| HOME | long_liquidation_cascade_watch | -1 | 0.009782 | -0.009782 |  |  |
| SNDK | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| ETH | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| BCH | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| BTC | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| PEPE | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| BSB | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| ZEC | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| BNB | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| XRP | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| TON | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| PIPPIN | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| DOGE | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| WLD | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| SOL | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| ALLO | mixed_liquidation_flow_watch | 0 | 0.014629 |  |  |  |
| ADA | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| MU | short_liquidation_squeeze_watch | 1 |  |  |  |  |

## Interpretation

This is price-only continuation labeling. It does not decide whether a liquidation event should be traded as continuation, reversal, or ignored without further regime and execution checks.
