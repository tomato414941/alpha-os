# Current OKX Liquidation Intensity

This joins recent OKX liquidation notional to OKX open interest. It is an event-intensity screen, not a trade instruction.

| asset | action | status | liq USD | OI USD | liq/OI | liq/vol | imbalance | score | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 192240 | 9154337 | 0.021000 | 0.002985 | -0.840002 | 217.2056 | label H long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| PIPPIN | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 98530 | 6547484 | 0.015048 | 0.000463 | -0.756385 | 157.0726 | label PIPPIN long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| WLD | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 506249 | 44063590 | 0.011489 | 0.000971 | -0.119952 | 120.8562 | label WLD mixed_liquidation_flow_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ALLO | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 64541 | 8066918 | 0.008001 | 0.000136 | -1.000000 | 86.8724 | label ALLO long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ZEC | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 140910 | 58476698 | 0.002410 | 0.000182 | 0.995406 | 31.3056 | label ZEC short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| MRVL | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 25143 | 11002551 | 0.002285 | 0.000231 | -1.000000 | 29.3188 | label MRVL long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| SOXL | long_liquidation_cascade_watch | low_liquidation_intensity_context | 2252 | 2969645 | 0.000758 | 0.000040 | -1.000000 | 12.9572 | keep SOXL as context only unless a fresh larger forced-flow event appears |
| BEAT | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 29715 | 42193012 | 0.000704 | 0.000039 | -0.681490 | 12.8974 | label BEAT long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ETH | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 130265 | 1184145775 | 0.000110 | 0.000015 | 0.948215 | 8.1299 | keep ETH as context only unless a fresh larger forced-flow event appears |
| MU | long_liquidation_cascade_watch | low_liquidation_intensity_context | 3847 | 25332179 | 0.000152 | 0.000013 | -1.000000 | 7.1163 | keep MU as context only unless a fresh larger forced-flow event appears |
| CBRS | long_liquidation_cascade_watch | low_liquidation_intensity_context | 235 | 1563924 | 0.000150 | 0.000004 | -1.000000 | 5.8779 | keep CBRS as context only unless a fresh larger forced-flow event appears |
| BSB | long_liquidation_cascade_watch | low_liquidation_intensity_context | 488 | 6293835 | 0.000078 | 0.000003 | -1.000000 | 5.4679 | keep BSB as context only unless a fresh larger forced-flow event appears |
| BTC | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 1687 | 1814221218 | 0.000001 | 0.000000 | 1.000000 | 5.2377 | keep BTC as context only unless a fresh larger forced-flow event appears |
| SOL | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 1397 | 186100935 | 0.000008 | 0.000002 | 1.000000 | 5.2243 | keep SOL as context only unless a fresh larger forced-flow event appears |
| NEAR | long_liquidation_cascade_watch | low_liquidation_intensity_context | 817 | 33266681 | 0.000025 | 0.000005 | -1.000000 | 5.1644 | keep NEAR as context only unless a fresh larger forced-flow event appears |
| HYPE | mixed_liquidation_flow_watch | low_liquidation_intensity_context | 7456 | 96259897 | 0.000077 | 0.000012 | -0.206083 | 5.0621 | keep HYPE as context only unless a fresh larger forced-flow event appears |
| CL | long_liquidation_cascade_watch | low_liquidation_intensity_context | 91 | 25802909 | 0.000004 | 0.000001 | -1.000000 | 3.9936 | keep CL as context only unless a fresh larger forced-flow event appears |
| BCH | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 23 | 22725533 | 0.000001 | 0.000000 | 1.000000 | 3.3789 | keep BCH as context only unless a fresh larger forced-flow event appears |
| DOGE | long_liquidation_cascade_watch | low_liquidation_intensity_context | 1 | 70814153 | 0.000000 | 0.000000 | -1.000000 | 2.0001 | keep DOGE as context only unless a fresh larger forced-flow event appears |

## Interpretation

High liquidation-to-OI rows are more likely to be true forced-flow events than rows that are merely large in dollar terms. The next test is still forward labeling with depth, fees, funding, and adverse-excursion checks.
