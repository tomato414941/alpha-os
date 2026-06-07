# Current OKX Liquidation Monitor Forward Label Summary

This labels repeated liquidation-monitor samples from each event timestamp. Positive continuation means price moved in the forced-flow direction over the horizon.

| asset | action | obs | cov15 | hit15 | mean cont15 | cov1h | hit1h | mean cont1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ONDO | short_liquidation_squeeze_watch | 3 | 3 | 1.0000 | 0.007452 | 3 | 1.0000 | 0.002006 |
| JTO | long_liquidation_cascade_watch | 3 | 3 | 1.0000 | 0.005121 | 3 | 1.0000 | 0.001120 |
| LTC | long_liquidation_cascade_watch | 3 | 3 | 1.0000 | 0.002599 | 3 | 1.0000 | 0.008268 |
| HYPE | long_liquidation_cascade_watch | 3 | 3 | 1.0000 | 0.002376 | 3 | 1.0000 | 0.008485 |
| LAB | short_liquidation_squeeze_watch | 3 | 3 | 1.0000 | 0.002311 | 3 | 1.0000 | 0.017641 |
| DOGE | long_liquidation_cascade_watch | 3 | 3 | 1.0000 | 0.001888 | 3 | 0.0000 | -0.000944 |
| H | short_liquidation_squeeze_watch | 3 | 3 | 1.0000 | 0.000240 | 3 | 1.0000 | 0.073497 |
| BEAT | short_liquidation_squeeze_watch | 3 | 3 | 0.0000 | -0.001452 | 3 | 1.0000 | 0.038382 |
| XLM | long_liquidation_cascade_watch | 3 | 3 | 0.0000 | -0.001476 | 3 | 1.0000 | 0.007870 |
| NEAR | short_liquidation_squeeze_watch | 3 | 3 | 0.0000 | -0.007367 | 3 | 1.0000 | 0.027996 |
| EDEN | long_liquidation_cascade_watch | 3 | 3 | 0.0000 | -0.008764 | 3 | 1.0000 | 0.032578 |
| ALLO | long_liquidation_cascade_watch | 3 | 3 | 0.0000 | -0.015074 | 3 | 0.0000 | -0.028189 |
| WLD | short_liquidation_squeeze_watch | 3 | 3 | 0.0000 | -0.026448 | 0 |  |  |
| BSB | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |
| BTC | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |
| ETH | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |
| HOME | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |
| OPN | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |
| ZEC | mixed_liquidation_flow_watch | 3 | 0 |  |  | 0 |  |  |

## Interpretation

This is still event-label evidence, not a trading decision. It should be joined with depth, fees, funding, and venue availability before sizing a paper trade.
