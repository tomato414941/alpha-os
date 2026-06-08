# Current OKX Liquidation Monitor Forward Label Summary

This labels repeated liquidation-monitor samples from each event timestamp. Positive continuation means price moved in the forced-flow direction over the horizon.

| asset | action | obs | cov15 | hit15 | mean cont15 | cov1h | hit1h | mean cont1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PEPE | long_liquidation_cascade_watch | 3 | 3 | 1.0000 | 0.001778 | 0 |  |  |
| XAU | short_liquidation_squeeze_watch | 3 | 3 | 1.0000 | 0.000161 | 0 |  |  |
| CL | long_liquidation_cascade_watch | 3 | 3 | 0.0000 | -0.000651 | 0 |  |  |
| BTC | short_liquidation_squeeze_watch | 3 | 3 | 0.0000 | -0.002486 | 0 |  |  |
| LAB | long_liquidation_cascade_watch | 3 | 3 | 0.0000 | -0.006542 | 0 |  |  |
| ZEC | short_liquidation_squeeze_watch | 3 | 1 | 0.0000 | -0.008025 | 0 |  |  |
| ALLO | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |
| BEAT | long_liquidation_cascade_watch | 3 | 0 |  |  | 0 |  |  |
| BSB | short_liquidation_squeeze_watch | 3 | 0 |  |  | 0 |  |  |
| ETH | short_liquidation_squeeze_watch | 3 | 0 |  |  | 0 |  |  |
| HOME | short_liquidation_squeeze_watch | 3 | 0 |  |  | 0 |  |  |
| MU | long_liquidation_cascade_watch | 3 | 0 |  |  | 0 |  |  |
| OPN | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |
| SUI | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |
| WLD | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |

## Interpretation

This is still event-label evidence, not a trading decision. It should be joined with depth, fees, funding, and venue availability before sizing a paper trade.
