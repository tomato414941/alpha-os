# Current OKX Liquidation Forward Labels

This labels liquidation-flow candidates with continuation returns. Positive continuation return means the forced-flow direction continued over that horizon.

| asset | action | dir | raw 15m | continuation 15m | raw 1h | continuation 1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FIL | short_liquidation_squeeze_watch | 1 | 0.009348 | 0.009348 |  |  |
| ZEC | long_liquidation_cascade_watch | -1 | -0.002797 | 0.002797 |  |  |
| XAU | short_liquidation_squeeze_watch | 1 | 0.001664 | 0.001664 |  |  |
| BCH | long_liquidation_cascade_watch | -1 | -0.000485 | 0.000485 |  |  |
| PEPE | short_liquidation_squeeze_watch | 1 | -0.000355 | -0.000355 |  |  |
| XRP | long_liquidation_cascade_watch | -1 | 0.000867 | -0.000867 |  |  |
| XAG | short_liquidation_squeeze_watch | 1 | -0.003212 | -0.003212 |  |  |
| LAB | short_liquidation_squeeze_watch | 1 | -0.003484 | -0.003484 |  |  |
| SNDK | short_liquidation_squeeze_watch | 1 | -0.004342 | -0.004342 | -0.001835 | -0.001835 |
| ONDO | long_liquidation_cascade_watch | -1 | 0.004457 | -0.004457 |  |  |
| TON | short_liquidation_squeeze_watch | 1 | -0.005760 | -0.005760 |  |  |
| CL | long_liquidation_cascade_watch | -1 | 0.007382 | -0.007382 |  |  |
| ADA | long_liquidation_cascade_watch | -1 | 0.009102 | -0.009102 |  |  |
| H | short_liquidation_squeeze_watch | 1 | -0.021125 | -0.021125 | -0.045856 | -0.045856 |
| ALLO | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| MU | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| PIPPIN | long_liquidation_cascade_watch | -1 |  |  |  |  |
| ETH | long_liquidation_cascade_watch | -1 |  |  |  |  |
| HYPE | long_liquidation_cascade_watch | -1 |  |  |  |  |
| WLD | long_liquidation_cascade_watch | -1 |  |  |  |  |
| BTC | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| SUI | long_liquidation_cascade_watch | -1 |  |  |  |  |
| BSB | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| BEAT | mixed_liquidation_flow_watch | 0 | -0.012750 |  |  |  |
| SOL | mixed_liquidation_flow_watch | 0 | 0.001204 |  |  |  |

## Interpretation

This is price-only continuation labeling. It does not decide whether a liquidation event should be traded as continuation, reversal, or ignored without further regime and execution checks.
