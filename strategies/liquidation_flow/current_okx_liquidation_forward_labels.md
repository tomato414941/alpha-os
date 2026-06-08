# Current OKX Liquidation Forward Labels

This labels liquidation-flow candidates with continuation returns. Positive continuation return means the forced-flow direction continued over that horizon.

| asset | action | dir | raw 15m | continuation 15m | raw 1h | continuation 1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PEPE | long_liquidation_cascade_watch | -1 | -0.001778 | 0.001778 |  |  |
| CL | long_liquidation_cascade_watch | -1 | 0.000651 | -0.000651 |  |  |
| BTC | short_liquidation_squeeze_watch | 1 | -0.002162 | -0.002162 |  |  |
| HYPE | long_liquidation_cascade_watch | -1 | 0.004583 | -0.004583 | 0.012052 | -0.012052 |
| LAB | long_liquidation_cascade_watch | -1 | 0.006230 | -0.006230 |  |  |
| ZEC | short_liquidation_squeeze_watch | 1 | -0.008958 | -0.008958 |  |  |
| XAU | short_liquidation_squeeze_watch | 1 | 0.000000 | 0.000000 |  |  |
| MU | long_liquidation_cascade_watch | -1 |  |  |  |  |
| BEAT | long_liquidation_cascade_watch | -1 |  |  |  |  |
| HOME | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| ETH | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| ALLO | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| WLD | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| BSB | short_liquidation_squeeze_watch | 1 |  |  |  |  |
| OPN | mixed_liquidation_flow_watch | 0 |  |  |  |  |
| SUI | mixed_liquidation_flow_watch | 0 | -0.001847 |  |  |  |

## Interpretation

This is price-only continuation labeling. It does not decide whether a liquidation event should be traded as continuation, reversal, or ignored without further regime and execution checks.
