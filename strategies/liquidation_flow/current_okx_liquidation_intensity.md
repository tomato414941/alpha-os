# Current OKX Liquidation Intensity

This joins recent OKX liquidation notional to OKX open interest. It is an event-intensity screen, not a trade instruction.

| asset | action | status | liq USD | OI USD | liq/OI | liq/vol | imbalance | score | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MRVL | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 117073 | 11540754 | 0.010144 | 0.001263 | 1.000000 | 108.6918 | label MRVL short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ALLO | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 97231 | 9833201 | 0.009888 | 0.000191 | -0.231131 | 104.3465 | label ALLO mixed_liquidation_flow_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BEAT | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 278278 | 41676529 | 0.006677 | 0.000327 | 0.007245 | 72.2306 | label BEAT mixed_liquidation_flow_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BSB | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 43377 | 7273143 | 0.005964 | 0.000124 | 1.000000 | 66.3291 | label BSB short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| WLD | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 121956 | 33116118 | 0.003683 | 0.000296 | -0.564928 | 43.0924 | label WLD long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| SNDK | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 33920 | 10798680 | 0.003141 | 0.000343 | 1.000000 | 38.0260 | label SNDK short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BCH | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 46269 | 22564454 | 0.002051 | 0.000527 | 1.000000 | 27.2775 | label BCH short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ZEC | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 84657 | 63456683 | 0.001334 | 0.000104 | 0.848938 | 20.0091 | label ZEC short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ETH | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 1259922 | 1199992514 | 0.001050 | 0.000144 | 0.930054 | 18.5279 | label ETH short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BTC | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 614341 | 1862851250 | 0.000330 | 0.000065 | 1.000000 | 11.1328 | keep BTC as context only unless a fresh larger forced-flow event appears |
| MU | long_liquidation_cascade_watch | low_liquidation_intensity_context | 13428 | 27532586 | 0.000488 | 0.000048 | -0.953448 | 10.9396 | keep MU as context only unless a fresh larger forced-flow event appears |
| PIPPIN | mixed_liquidation_flow_watch | low_liquidation_intensity_context | 2662 | 4598549 | 0.000579 | 0.000023 | -0.008532 | 9.2322 | keep PIPPIN as context only unless a fresh larger forced-flow event appears |
| XRP | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 9249 | 82014513 | 0.000113 | 0.000046 | 1.000000 | 7.1205 | keep XRP as context only unless a fresh larger forced-flow event appears |
| XAU | long_liquidation_cascade_watch | low_liquidation_intensity_context | 9769 | 102311498 | 0.000095 | 0.000042 | -1.000000 | 6.9705 | keep XAU as context only unless a fresh larger forced-flow event appears |
| BNB | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 5291 | 45267109 | 0.000117 | 0.000062 | 1.000000 | 6.9218 | keep BNB as context only unless a fresh larger forced-flow event appears |
| DOGE | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 6224 | 71615783 | 0.000087 | 0.000018 | 1.000000 | 6.6792 | keep DOGE as context only unless a fresh larger forced-flow event appears |
| XAG | long_liquidation_cascade_watch | low_liquidation_intensity_context | 2450 | 20997438 | 0.000117 | 0.000019 | -1.000000 | 6.5705 | keep XAG as context only unless a fresh larger forced-flow event appears |
| HYPE | long_liquidation_cascade_watch | low_liquidation_intensity_context | 12115 | 99344974 | 0.000122 | 0.000021 | -0.570949 | 6.4555 | keep HYPE as context only unless a fresh larger forced-flow event appears |
| SOL | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 4274 | 186981208 | 0.000023 | 0.000006 | 1.000000 | 5.8680 | keep SOL as context only unless a fresh larger forced-flow event appears |
| TON | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 1209 | 16040417 | 0.000075 | 0.000025 | 0.846869 | 5.5429 | keep TON as context only unless a fresh larger forced-flow event appears |
| CL | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 231 | 25071768 | 0.000009 | 0.000001 | 1.000000 | 4.4582 | keep CL as context only unless a fresh larger forced-flow event appears |
| NEAR | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 100 | 32164400 | 0.000003 | 0.000001 | 1.000000 | 4.0320 | keep NEAR as context only unless a fresh larger forced-flow event appears |
| ONDO | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 92 | 13501437 | 0.000007 | 0.000002 | 1.000000 | 4.0319 | keep ONDO as context only unless a fresh larger forced-flow event appears |
| HOME | long_liquidation_cascade_watch | low_liquidation_intensity_context | 30 | 3084765 | 0.000010 | 0.000001 | -1.000000 | 3.5746 | keep HOME as context only unless a fresh larger forced-flow event appears |
| FIL | long_liquidation_cascade_watch | low_liquidation_intensity_context | 26 | 11767052 | 0.000002 | 0.000000 | -1.000000 | 3.4447 | keep FIL as context only unless a fresh larger forced-flow event appears |

## Interpretation

High liquidation-to-OI rows are more likely to be true forced-flow events than rows that are merely large in dollar terms. The next test is still forward labeling with depth, fees, funding, and adverse-excursion checks.
