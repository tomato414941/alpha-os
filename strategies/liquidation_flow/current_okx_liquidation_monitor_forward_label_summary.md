# Current OKX Liquidation Monitor Forward Label Summary

This labels repeated liquidation-monitor samples from each event timestamp. Positive continuation means price moved in the forced-flow direction over the horizon.

| asset | action | obs | cov15 | hit15 | mean cont15 | cov1h | hit1h | mean cont1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JTO | long_liquidation_cascade_watch | 3 | 3 | 1.0000 | 0.010562 | 0 |  |  |
| ONDO | short_liquidation_squeeze_watch | 3 | 3 | 1.0000 | 0.007452 | 0 |  |  |
| H | short_liquidation_squeeze_watch | 3 | 3 | 1.0000 | 0.006116 | 0 |  |  |
| DOGE | long_liquidation_cascade_watch | 3 | 3 | 1.0000 | 0.003185 | 0 |  |  |
| LAB | short_liquidation_squeeze_watch | 3 | 3 | 1.0000 | 0.002773 | 0 |  |  |
| LTC | long_liquidation_cascade_watch | 3 | 3 | 1.0000 | 0.002599 | 0 |  |  |
| HYPE | long_liquidation_cascade_watch | 3 | 3 | 1.0000 | 0.002376 | 3 | 1.0000 | 0.012218 |
| BEAT | short_liquidation_squeeze_watch | 3 | 3 | 1.0000 | 0.000968 | 0 |  |  |
| XLM | long_liquidation_cascade_watch | 3 | 3 | 0.0000 | -0.001476 | 0 |  |  |
| EDEN | long_liquidation_cascade_watch | 3 | 3 | 0.0000 | -0.008764 | 0 |  |  |
| ALLO | long_liquidation_cascade_watch | 3 | 3 | 0.0000 | -0.015074 | 0 |  |  |
| NEAR | short_liquidation_squeeze_watch | 3 | 3 | 0.0000 | -0.019155 | 0 |  |  |
| BSB | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |
| BTC | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |
| ETH | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |
| HOME | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |
| OPN | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |
| WLD | short_liquidation_squeeze_watch | 3 | 0 |  |  | 0 |  |  |
| ZEC | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |

## Interpretation

This is still event-label evidence, not a trading decision. It should be joined with depth, fees, funding, and venue availability before sizing a paper trade.
