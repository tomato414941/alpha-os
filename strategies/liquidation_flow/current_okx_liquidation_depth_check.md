# Current OKX Liquidation Depth Check

This checks visible OKX book depth for liquidation-monitor candidates. It is not a fill guarantee.

| asset | action | spread bps | bid depth 5bps | ask depth 5bps | bid depth 10bps | ask depth 10bps | monitor score | depth score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LTC | long_liquidation_cascade_watch | 2.3824 | 33473 | 26234 | 92199 | 95779 | 0.035068 | 0.029440 |
| ONDO | short_liquidation_squeeze_watch | 2.8806 | 14318 | 13968 | 29835 | 26141 | 0.016723 | 0.011611 |
| ZEC | mixed_liquidation_flow_watch | 0.2362 | 16507 | 56008 | 82299 | 135785 | 0.050522 | 0.008747 |
| JTO | long_liquidation_cascade_watch | 1.6149 | 2916 | 2208 | 9749 | 7819 | 0.094599 | 0.003139 |
| H | short_liquidation_squeeze_watch | 1.7710 | 1565 | 767 | 8137 | 4446 | 0.021281 | 0.002300 |
| XLM | long_liquidation_cascade_watch | 4.9444 | 11465 | 532 | 50850 | 27992 | 0.023186 | 0.000753 |
| BEAT | short_liquidation_squeeze_watch | 2.1265 | 846 | 1778 | 9304 | 7679 | 0.127525 | 0.000291 |
| WLD | short_liquidation_squeeze_watch | 2.0728 | 2430 | 2900 | 19616 | 10034 | 0.117841 | 0.000265 |
| BSB | mixed_liquidation_flow_watch | 2.7728 | 1968 | 1230 | 3406 | 4948 | 0.109096 | 0.000081 |
| HOME | mixed_liquidation_flow_watch | 6.5898 | 85 | 197 | 373 | 993 | 0.018062 | 0.000014 |

## Interpretation

The useful follow-up candidates are liquidation signals that persist and have enough visible depth near touch. This still excludes account fees, hidden liquidity, maker fill probability, and slippage during a fast liquidation event.
