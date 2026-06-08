# Current OKX Liquidation Depth Check

This checks visible OKX book depth for liquidation-monitor candidates. It is not a fill guarantee.

| asset | action | spread bps | bid depth 5bps | ask depth 5bps | bid depth 10bps | ask depth 10bps | monitor score | depth score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XAU | short_liquidation_squeeze_watch | 0.2308 | 1602933 | 1214830 | 2365279 | 2439601 | 0.113368 | 0.982461 |
| ETH | short_liquidation_squeeze_watch | 0.0595 | 947889 | 1928078 | 947889 | 1928078 | 0.027553 | 0.551069 |
| BTC | short_liquidation_squeeze_watch | 0.0158 | 358952 | 920551 | 358952 | 920551 | 0.024579 | 0.447470 |
| MU | long_liquidation_cascade_watch | 0.1074 | 15883 | 96701 | 68991 | 115021 | 0.261991 | 0.192321 |
| ZEC | short_liquidation_squeeze_watch | 0.2378 | 29648 | 74452 | 104423 | 150481 | 0.015398 | 0.063088 |
| ALLO | mixed_liquidation_flow_watch | 0.2321 | 1202 | 2651 | 5521 | 8956 | 0.022347 | 0.003770 |
| BEAT | long_liquidation_cascade_watch | 0.2333 | 1243 | 6731 | 7966 | 18442 | 0.069987 | 0.001563 |
| WLD | mixed_liquidation_flow_watch | 2.1666 | 20295 | 5550 | 57934 | 38233 | 0.017785 | 0.001341 |
| HOME | short_liquidation_squeeze_watch | 3.1751 | 699 | 1336 | 1246 | 2861 | 0.047642 | 0.001186 |
| BSB | short_liquidation_squeeze_watch | 3.0614 | 1926 | 2206 | 6290 | 4762 | 0.015270 | 0.000540 |

## Interpretation

The useful follow-up candidates are liquidation signals that persist and have enough visible depth near touch. This still excludes account fees, hidden liquidity, maker fill probability, and slippage during a fast liquidation event.
