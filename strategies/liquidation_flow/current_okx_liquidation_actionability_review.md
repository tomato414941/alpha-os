# Current OKX Liquidation Actionability Review

This joins liquidation persistence, monitor-sample continuation labels, and visible near-touch depth. It is a triage view, not an order plan.

| asset | action | obs | monitor score | cont15 | spread bps | near depth 5bps | score | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| XAU | short_liquidation_squeeze_watch | 3 | 0.113368 | 0.000161 | 0.2319 | 1019894 | 4.116588 | first checks support follow-up |
| MU | long_liquidation_cascade_watch | 3 | 0.261991 |  | 0.1144 | 11068 | 0.180995 | waiting for matching forward label |
| ETH | short_liquidation_squeeze_watch | 3 | 0.027553 |  | 0.0600 | 1176932 | 0.063777 | waiting for matching forward label |
| WLD | mixed_liquidation_flow_watch | 3 | 0.017785 |  | 2.0918 | 13729 | 0.058892 | waiting for matching forward label |
| BEAT | long_liquidation_cascade_watch | 3 | 0.069987 |  | 5.0484 | 4286 | 0.056424 | waiting for matching forward label |
| PEPE | long_liquidation_cascade_watch | 3 | 0.000060 | 0.001778 |  |  | 0.035622 | first checks support follow-up |
| HOME | short_liquidation_squeeze_watch | 3 | 0.047642 |  | 6.0864 | 191 | 0.024774 | waiting for matching forward label |
| ALLO | mixed_liquidation_flow_watch | 3 | 0.022347 |  | 0.2612 | 980 | 0.016075 | waiting for matching forward label |
| BSB | short_liquidation_squeeze_watch | 3 | 0.015270 |  | 3.3328 | 985 | 0.012558 | waiting for matching forward label |
| BTC | short_liquidation_squeeze_watch | 3 | 0.024579 | -0.002486 | 0.0159 | 143490 | 0.002458 | continuation label weak |
| OPN | mixed_liquidation_flow_watch | 3 | 0.004453 |  |  |  | 0.002227 | waiting for matching forward label |
| ZEC | short_liquidation_squeeze_watch | 3 | 0.015398 | -0.008025 | 0.2346 | 22370 | 0.001540 | continuation label weak |
| SUI | mixed_liquidation_flow_watch | 3 | 0.002841 |  |  |  | 0.001421 | waiting for matching forward label |
| CL | long_liquidation_cascade_watch | 3 | 0.010592 | -0.000651 |  |  | 0.001059 | continuation label weak |
| LAB | long_liquidation_cascade_watch | 3 | 0.005421 | -0.006542 |  |  | 0.000542 | continuation label weak |

## Interpretation

A high score means the candidate has some combination of persistent liquidation flow, positive monitor-sample continuation, and visible depth. Thin-depth high-signal names should be treated as small-size probes until better venue depth is found.
