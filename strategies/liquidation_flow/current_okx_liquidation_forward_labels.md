# Current OKX Liquidation Forward Labels

This labels liquidation-flow candidates with continuation returns. Positive continuation return means the forced-flow direction continued over that horizon.

| asset | action | dir | raw 15m | continuation 15m | raw 1h | continuation 1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BCH | long_liquidation_cascade_watch | -1 | -0.007786 | 0.007786 |  |  |
| PEPE | long_liquidation_cascade_watch | -1 | -0.006857 | 0.006857 |  |  |
| ADA | long_liquidation_cascade_watch | -1 | -0.004189 | 0.004189 |  |  |
| MU | long_liquidation_cascade_watch | -1 | -0.004002 | 0.004002 |  |  |
| NEAR | long_liquidation_cascade_watch | -1 | -0.003833 | 0.003833 |  |  |
| SNDK | long_liquidation_cascade_watch | -1 | -0.003794 | 0.003794 |  |  |
| FIL | long_liquidation_cascade_watch | -1 | -0.003154 | 0.003154 |  |  |
| BNB | long_liquidation_cascade_watch | -1 | -0.001679 | 0.001679 |  |  |
| XAU | long_liquidation_cascade_watch | -1 | 0.000023 | -0.000023 |  |  |
| XAG | long_liquidation_cascade_watch | -1 | 0.001330 | -0.001330 |  |  |
| MRVL | long_liquidation_cascade_watch | -1 | 0.002454 | -0.002454 |  |  |
| ALLO | short_liquidation_squeeze_watch | 1 | -0.011354 | -0.011354 |  |  |
| SUI | long_liquidation_cascade_watch | -1 |  |  |  |  |
| SOL | long_liquidation_cascade_watch | -1 |  |  |  |  |
| BTC | long_liquidation_cascade_watch | -1 |  |  |  |  |
| DOGE | long_liquidation_cascade_watch | -1 |  |  |  |  |
| H | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| ETH | long_liquidation_cascade_watch | -1 |  |  |  |  |
| PIPPIN | long_liquidation_cascade_watch | -1 |  |  |  |  |
| ZEC | long_liquidation_cascade_watch | -1 |  |  |  |  |
| WLD | long_liquidation_cascade_watch | -1 |  |  |  |  |
| XRP | long_liquidation_cascade_watch | -1 |  |  |  |  |
| HYPE | long_liquidation_cascade_watch | -1 |  |  |  |  |
| BSB | long_liquidation_cascade_watch | -1 |  |  |  |  |
| LAB | long_liquidation_cascade_watch | -1 |  |  |  |  |

## Interpretation

This is price-only continuation labeling. It does not decide whether a liquidation event should be traded as continuation, reversal, or ignored without further regime and execution checks.
