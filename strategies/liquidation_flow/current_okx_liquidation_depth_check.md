# Current OKX Liquidation Depth Check

This checks visible OKX book depth for liquidation-monitor candidates. It is not a fill guarantee.

| asset | action | spread bps | bid depth 5bps | ask depth 5bps | bid depth 10bps | ask depth 10bps | monitor score | depth score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XAU | short_liquidation_squeeze_watch | 0.2319 | 1066747 | 1019894 | 1807852 | 1982492 | 0.113368 | 0.977587 |
| ETH | short_liquidation_squeeze_watch | 0.0600 | 1748063 | 1176932 | 1748063 | 1176932 | 0.027553 | 0.551069 |
| BTC | short_liquidation_squeeze_watch | 0.0159 | 769775 | 143490 | 769775 | 143490 | 0.024579 | 0.178875 |
| MU | long_liquidation_cascade_watch | 0.1144 | 29133 | 11068 | 85824 | 42313 | 0.261991 | 0.125858 |
| ZEC | short_liquidation_squeeze_watch | 0.2346 | 22370 | 23495 | 81723 | 105302 | 0.015398 | 0.048257 |
| WLD | mixed_liquidation_flow_watch | 2.0918 | 13994 | 13729 | 51506 | 65243 | 0.017785 | 0.003436 |
| ALLO | mixed_liquidation_flow_watch | 0.2612 | 3586 | 980 | 9006 | 5241 | 0.022347 | 0.002732 |
| BSB | short_liquidation_squeeze_watch | 3.3328 | 4283 | 985 | 8264 | 3603 | 0.015270 | 0.000254 |
| BEAT | long_liquidation_cascade_watch | 5.0484 | 4574 | 4286 | 31342 | 14236 | 0.069987 | 0.000249 |
| HOME | short_liquidation_squeeze_watch | 6.0864 | 726 | 191 | 2463 | 1197 | 0.047642 | 0.000169 |

## Interpretation

The useful follow-up candidates are liquidation signals that persist and have enough visible depth near touch. This still excludes account fees, hidden liquidity, maker fill probability, and slippage during a fast liquidation event.
