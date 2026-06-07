# Current OKX Liquidation Actionability Review

This joins liquidation persistence, monitor-sample continuation labels, and visible near-touch depth. It is a triage view, not an order plan.

| asset | action | obs | monitor score | cont15 | spread bps | near depth 5bps | score | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LTC | long_liquidation_cascade_watch | 3 | 0.035068 | 0.002599 | 2.3824 | 26234 | 0.926541 | first checks support follow-up |
| ONDO | short_liquidation_squeeze_watch | 3 | 0.016723 | 0.007452 | 2.8806 | 13968 | 0.650667 | first checks support follow-up |
| JTO | long_liquidation_cascade_watch | 3 | 0.094599 | 0.010562 | 1.6149 | 2208 | 0.442550 | signal ok but visible depth thin |
| H | short_liquidation_squeeze_watch | 3 | 0.021281 | 0.006116 | 1.7710 | 767 | 0.186923 | signal ok but visible depth thin |
| BEAT | short_liquidation_squeeze_watch | 3 | 0.127525 | 0.000968 | 2.1265 | 846 | 0.186678 | signal ok but visible depth thin |
| DOGE | long_liquidation_cascade_watch | 3 | 0.015966 | 0.003185 |  |  | 0.079676 | first checks support follow-up |
| ZEC | mixed_liquidation_flow_watch | 3 | 0.050522 |  | 0.2362 | 16507 | 0.075261 | waiting for matching forward label |
| WLD | short_liquidation_squeeze_watch | 3 | 0.117841 |  | 2.0728 | 2430 | 0.071071 | waiting for matching forward label |
| LAB | short_liquidation_squeeze_watch | 3 | 0.006273 | 0.002773 |  |  | 0.061739 | first checks support follow-up |
| BSB | mixed_liquidation_flow_watch | 3 | 0.109096 |  | 2.7728 | 1230 | 0.060700 | waiting for matching forward label |
| HYPE | long_liquidation_cascade_watch | 3 | 0.006522 | 0.002376 |  |  | 0.054036 | first checks support follow-up |
| HOME | mixed_liquidation_flow_watch | 3 | 0.018062 |  | 6.5898 | 85 | 0.009456 | waiting for matching forward label |
| ETH | mixed_liquidation_flow_watch | 3 | 0.008588 |  |  |  | 0.004294 | waiting for matching forward label |
| BTC | mixed_liquidation_flow_watch | 3 | 0.005563 |  |  |  | 0.002781 | waiting for matching forward label |
| XLM | long_liquidation_cascade_watch | 3 | 0.023186 | -0.001476 | 4.9444 | 532 | 0.002319 | continuation label weak |
| NEAR | short_liquidation_squeeze_watch | 3 | 0.010032 | -0.019155 |  |  | 0.001003 | continuation label weak |
| OPN | mixed_liquidation_flow_watch | 3 | 0.000999 |  |  |  | 0.000500 | waiting for matching forward label |
| ALLO | long_liquidation_cascade_watch | 3 | 0.004683 | -0.015074 |  |  | 0.000468 | continuation label weak |
| EDEN | long_liquidation_cascade_watch | 3 | 0.000245 | -0.008764 |  |  | 0.000025 | continuation label weak |

## Interpretation

A high score means the candidate has some combination of persistent liquidation flow, positive monitor-sample continuation, and visible depth. Thin-depth high-signal names should be treated as small-size probes until better venue depth is found.
