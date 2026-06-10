# Current OKX Liquidation Depth Check

This checks visible OKX book depth for liquidation-monitor candidates. It is not a fill guarantee.

| asset | action | spread bps | bid depth 5bps | ask depth 5bps | bid depth 10bps | ask depth 10bps | monitor score | depth score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XAU | short_liquidation_squeeze_watch | 0.2314 | 1164969 | 998335 | 2288908 | 1918117 | 0.113368 | 0.979809 |
| ETH | short_liquidation_squeeze_watch | 0.0602 | 1289604 | 1201896 | 1289604 | 1201896 | 0.027553 | 0.551069 |
| BTC | short_liquidation_squeeze_watch | 0.0160 | 618393 | 484882 | 618393 | 484882 | 0.024579 | 0.491570 |
| MU | long_liquidation_cascade_watch | 0.1066 | 17778 | 30154 | 45865 | 57151 | 0.261991 | 0.217037 |
| ZEC | short_liquidation_squeeze_watch | 0.2231 | 25685 | 53830 | 106307 | 125613 | 0.015398 | 0.058273 |
| BEAT | long_liquidation_cascade_watch | 0.2269 | 5023 | 2601 | 14052 | 10894 | 0.069987 | 0.003365 |
| ALLO | mixed_liquidation_flow_watch | 0.2392 | 2537 | 962 | 9060 | 9172 | 0.022347 | 0.002928 |
| WLD | mixed_liquidation_flow_watch | 2.0360 | 13856 | 9704 | 46008 | 34443 | 0.017785 | 0.002495 |
| BSB | short_liquidation_squeeze_watch | 3.4276 | 93 | 441 | 3029 | 4878 | 0.015270 | 0.000023 |
| HOME | short_liquidation_squeeze_watch | 3.5404 | 234 | 11 | 1321 | 1100 | 0.047642 | 0.000017 |

## Interpretation

The useful follow-up candidates are liquidation signals that persist and have enough visible depth near touch. This still excludes account fees, hidden liquidity, maker fill probability, and slippage during a fast liquidation event.
