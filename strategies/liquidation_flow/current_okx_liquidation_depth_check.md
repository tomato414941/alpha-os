# Current OKX Liquidation Depth Check

This checks visible OKX book depth for liquidation-monitor candidates. It is not a fill guarantee.

| asset | action | spread bps | bid depth 5bps | ask depth 5bps | bid depth 10bps | ask depth 10bps | monitor score | depth score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XAU | short_liquidation_squeeze_watch | 0.2308 | 2150947 | 1280200 | 3306048 | 2426718 | 0.113368 | 0.982257 |
| ETH | short_liquidation_squeeze_watch | 0.0594 | 1139366 | 1188207 | 1139366 | 1188207 | 0.027553 | 0.551069 |
| BTC | short_liquidation_squeeze_watch | 0.0158 | 899200 | 526359 | 899200 | 526359 | 0.024579 | 0.491570 |
| MU | long_liquidation_cascade_watch | 0.1059 | 22933 | 34486 | 93366 | 60766 | 0.261991 | 0.281630 |
| ZEC | short_liquidation_squeeze_watch | 0.2194 | 34792 | 48217 | 150670 | 172664 | 0.015398 | 0.080231 |
| BEAT | long_liquidation_cascade_watch | 0.2279 | 7020 | 27525 | 18422 | 33235 | 0.069987 | 0.009042 |
| ALLO | mixed_liquidation_flow_watch | 0.2534 | 1503 | 1820 | 5371 | 8624 | 0.022347 | 0.004317 |
| WLD | mixed_liquidation_flow_watch | 1.8287 | 19258 | 15055 | 37819 | 36306 | 0.017785 | 0.004310 |
| BSB | short_liquidation_squeeze_watch | 3.1226 | 1543 | 2098 | 3066 | 4220 | 0.015270 | 0.000424 |
| HOME | short_liquidation_squeeze_watch | 3.5392 | 113 | 102 | 974 | 1606 | 0.047642 | 0.000155 |

## Interpretation

The useful follow-up candidates are liquidation signals that persist and have enough visible depth near touch. This still excludes account fees, hidden liquidity, maker fill probability, and slippage during a fast liquidation event.
