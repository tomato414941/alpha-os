# Current OKX Liquidation Depth Check

This checks visible OKX book depth for liquidation-monitor candidates. It is not a fill guarantee.

| asset | action | spread bps | bid depth 5bps | ask depth 5bps | bid depth 10bps | ask depth 10bps | monitor score | depth score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XAU | short_liquidation_squeeze_watch | 0.2308 | 1699476 | 1213655 | 2617732 | 2130702 | 0.113368 | 0.982189 |
| ETH | short_liquidation_squeeze_watch | 0.0587 | 1019968 | 1214759 | 1019968 | 1214759 | 0.027553 | 0.551069 |
| BTC | short_liquidation_squeeze_watch | 0.0156 | 409894 | 897792 | 409894 | 897792 | 0.024579 | 0.491570 |
| MU | long_liquidation_cascade_watch | 0.1046 | 20013 | 76383 | 119265 | 106564 | 0.261991 | 0.248897 |
| ZEC | short_liquidation_squeeze_watch | 0.2253 | 51200 | 25589 | 156770 | 102821 | 0.015398 | 0.057481 |
| BEAT | long_liquidation_cascade_watch | 0.2282 | 3549 | 10038 | 15943 | 16749 | 0.069987 | 0.004565 |
| WLD | mixed_liquidation_flow_watch | 2.0549 | 11279 | 12892 | 32617 | 37575 | 0.017785 | 0.002873 |
| HOME | short_liquidation_squeeze_watch | 3.2118 | 1360 | 1302 | 1892 | 1704 | 0.047642 | 0.002185 |
| BSB | short_liquidation_squeeze_watch | 2.9485 | 2465 | 3745 | 3685 | 4801 | 0.015270 | 0.000718 |
| ALLO | mixed_liquidation_flow_watch | 2.6663 | 2943 | 1904 | 6383 | 8710 | 0.022347 | 0.000520 |

## Interpretation

The useful follow-up candidates are liquidation signals that persist and have enough visible depth near touch. This still excludes account fees, hidden liquidity, maker fill probability, and slippage during a fast liquidation event.
