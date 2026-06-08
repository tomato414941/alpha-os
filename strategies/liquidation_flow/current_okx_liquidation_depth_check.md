# Current OKX Liquidation Depth Check

This checks visible OKX book depth for liquidation-monitor candidates. It is not a fill guarantee.

| asset | action | spread bps | bid depth 5bps | ask depth 5bps | bid depth 10bps | ask depth 10bps | monitor score | depth score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XAU | short_liquidation_squeeze_watch | 0.2306 | 1062653 | 1920702 | 2245891 | 3303592 | 0.113368 | 0.983436 |
| MU | long_liquidation_cascade_watch | 0.1057 | 57256 | 101669 | 95779 | 182775 | 0.261991 | 0.704925 |
| ETH | short_liquidation_squeeze_watch | 0.0590 | 1294732 | 1196107 | 1294732 | 1196107 | 0.027553 | 0.551069 |
| BTC | short_liquidation_squeeze_watch | 0.0156 | 947089 | 545043 | 947089 | 545043 | 0.024579 | 0.491570 |
| ZEC | short_liquidation_squeeze_watch | 0.2260 | 36208 | 13490 | 76744 | 54076 | 0.015398 | 0.030206 |
| BEAT | long_liquidation_cascade_watch | 0.2285 | 3632 | 3590 | 9930 | 23695 | 0.069987 | 0.004613 |
| ALLO | mixed_liquidation_flow_watch | 0.2230 | 2008 | 1301 | 7586 | 5738 | 0.022347 | 0.004245 |
| WLD | mixed_liquidation_flow_watch | 2.0866 | 12761 | 17271 | 43574 | 52857 | 0.017785 | 0.003201 |
| HOME | short_liquidation_squeeze_watch | 3.2222 | 847 | 565 | 1476 | 2252 | 0.047642 | 0.000945 |
| BSB | short_liquidation_squeeze_watch | 2.9954 | 10645 | 2485 | 12167 | 4490 | 0.015270 | 0.000712 |

## Interpretation

The useful follow-up candidates are liquidation signals that persist and have enough visible depth near touch. This still excludes account fees, hidden liquidity, maker fill probability, and slippage during a fast liquidation event.
