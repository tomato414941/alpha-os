# Current OKX Liquidation Forward Labels

This labels liquidation-flow candidates with continuation returns. Positive continuation return means the forced-flow direction continued over that horizon.

| asset | action | dir | raw 15m | continuation 15m | raw 1h | continuation 1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TON | short_liquidation_squeeze_watch | 1 | 0.005294 | 0.005294 |  |  |
| SNDK | long_liquidation_cascade_watch | -1 | -0.003548 | 0.003548 |  |  |
| MU | long_liquidation_cascade_watch | -1 | -0.001546 | 0.001546 |  |  |
| CL | short_liquidation_squeeze_watch | 1 | 0.000854 | 0.000854 |  |  |
| XAU | long_liquidation_cascade_watch | -1 | 0.000859 | -0.000859 |  |  |
| XRP | long_liquidation_cascade_watch | -1 | 0.002648 | -0.002648 |  |  |
| DOGE | long_liquidation_cascade_watch | -1 | 0.002958 | -0.002958 |  |  |
| PEPE | long_liquidation_cascade_watch | -1 | 0.003272 | -0.003272 |  |  |
| SOL | long_liquidation_cascade_watch | -1 | 0.003536 | -0.003536 |  |  |
| MRVL | long_liquidation_cascade_watch | -1 | 0.004745 | -0.004745 |  |  |
| SUI | long_liquidation_cascade_watch | -1 | 0.005746 | -0.005746 |  |  |
| XAG | long_liquidation_cascade_watch | -1 |  |  |  |  |
| BEAT | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| HYPE | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| BSB | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| PIPPIN | long_liquidation_cascade_watch | -1 |  |  |  |  |
| WLD | long_liquidation_cascade_watch | -1 |  |  |  |  |
| ALLO | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| NEAR | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| LAB | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| BTC | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| ETH | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| HOME | mixed_liquidation_flow_watch | 0 | -0.001520 |  |  |  |
| JTO | long_liquidation_cascade_watch | -1 |  |  |  |  |
| ZEC | mixed_liquidation_flow_watch | 0 |  |  |  |  |

## Interpretation

This is price-only continuation labeling. It does not decide whether a liquidation event should be traded as continuation, reversal, or ignored without further regime and execution checks.
