# Current OKX Liquidation Depth Check

This checks visible OKX book depth for liquidation-monitor candidates. It is not a fill guarantee.

| asset | action | spread bps | bid depth 5bps | ask depth 5bps | bid depth 10bps | ask depth 10bps | monitor score | depth score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XAU | short_liquidation_squeeze_watch | 0.2300 | 1043633 | 1271560 | 1887060 | 1618238 | 0.113368 | 0.985931 |
| ETH | short_liquidation_squeeze_watch | 0.0594 | 590417 | 1002205 | 590417 | 1002205 | 0.027553 | 0.551069 |
| BTC | short_liquidation_squeeze_watch | 0.0158 | 537296 | 182429 | 537296 | 182429 | 0.024579 | 0.227416 |
| ZEC | short_liquidation_squeeze_watch | 0.2292 | 14794 | 40879 | 79384 | 88222 | 0.015398 | 0.032663 |
| MU | long_liquidation_cascade_watch | 2.3528 | 16525 | 18303 | 39389 | 38972 | 0.261991 | 0.009137 |
| ALLO | mixed_liquidation_flow_watch | 0.3061 | 1150 | 1618 | 5488 | 6826 | 0.022347 | 0.002734 |
| WLD | mixed_liquidation_flow_watch | 2.1261 | 11044 | 8017 | 49033 | 36730 | 0.017785 | 0.001974 |
| BSB | short_liquidation_squeeze_watch | 3.0161 | 928 | 1410 | 1975 | 2793 | 0.015270 | 0.000264 |
| BEAT | long_liquidation_cascade_watch | 2.0376 | 1614 | 1993 | 7038 | 5534 | 0.069987 | 0.000232 |
| HOME | short_liquidation_squeeze_watch | 3.4194 | 494 | 18 | 3618 | 2637 | 0.047642 | 0.000028 |

## Interpretation

The useful follow-up candidates are liquidation signals that persist and have enough visible depth near touch. This still excludes account fees, hidden liquidity, maker fill probability, and slippage during a fast liquidation event.
