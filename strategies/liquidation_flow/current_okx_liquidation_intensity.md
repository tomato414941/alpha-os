# Current OKX Liquidation Intensity

This joins recent OKX liquidation notional to OKX open interest. It is an event-intensity screen, not a trade instruction.

| asset | action | status | liq USD | OI USD | liq/OI | liq/vol | imbalance | score | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ALLO | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 406425 | 9809208 | 0.041433 | 0.000929 | -0.339652 | 420.6765 | label ALLO mixed_liquidation_flow_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ZEC | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 268327 | 61226438 | 0.004383 | 0.000313 | -0.970942 | 51.2893 | label ZEC long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| PIPPIN | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 17548 | 4346853 | 0.004037 | 0.000171 | -0.870308 | 46.4037 | label PIPPIN long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BEAT | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 132070 | 40292031 | 0.003278 | 0.000164 | -0.224351 | 38.3625 | label BEAT mixed_liquidation_flow_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BSB | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 22296 | 7655485 | 0.002912 | 0.000060 | -0.495379 | 34.4804 | label BSB mixed_liquidation_flow_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| TON | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 35445 | 15902197 | 0.002229 | 0.000707 | 1.000000 | 28.9596 | label TON short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ETH | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 1564706 | 1186306165 | 0.001319 | 0.000182 | -0.575043 | 20.5823 | label ETH long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BTC | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 2032870 | 1814678344 | 0.001120 | 0.000223 | -0.280142 | 18.0971 | label BTC mixed_liquidation_flow_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ONDO | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 15566 | 13779682 | 0.001130 | 0.000327 | -1.000000 | 17.5642 | label ONDO long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| WLD | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 33344 | 32770882 | 0.001017 | 0.000081 | -0.683248 | 16.0922 | label WLD long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| MU | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 24448 | 36436012 | 0.000671 | 0.000127 | 1.000000 | 13.1475 | label MU short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| PEPE | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 8455 | 17152211 | 0.000493 | 0.000068 | 0.927603 | 10.7420 | keep PEPE as context only unless a fresh larger forced-flow event appears |
| HYPE | long_liquidation_cascade_watch | low_liquidation_intensity_context | 23849 | 87470954 | 0.000273 | 0.000050 | -0.913851 | 8.9601 | keep HYPE as context only unless a fresh larger forced-flow event appears |
| SOL | mixed_liquidation_flow_watch | low_liquidation_intensity_context | 57397 | 181208875 | 0.000317 | 0.000081 | 0.338877 | 8.6186 | keep SOL as context only unless a fresh larger forced-flow event appears |
| DOGE | mixed_liquidation_flow_watch | low_liquidation_intensity_context | 13229 | 71242462 | 0.000186 | 0.000040 | 0.369646 | 6.7273 | keep DOGE as context only unless a fresh larger forced-flow event appears |
| SUI | long_liquidation_cascade_watch | low_liquidation_intensity_context | 3021 | 20615561 | 0.000147 | 0.000037 | -0.867958 | 6.6996 | keep SUI as context only unless a fresh larger forced-flow event appears |
| XAU | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 5380 | 101911124 | 0.000053 | 0.000026 | 1.000000 | 6.2777 | keep XAU as context only unless a fresh larger forced-flow event appears |
| H | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 1408 | 17080894 | 0.000082 | 0.000019 | 1.000000 | 5.9864 | keep H as context only unless a fresh larger forced-flow event appears |
| CL | long_liquidation_cascade_watch | low_liquidation_intensity_context | 924 | 24483595 | 0.000038 | 0.000006 | -1.000000 | 5.3505 | keep CL as context only unless a fresh larger forced-flow event appears |
| LAB | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 643 | 17934729 | 0.000036 | 0.000004 | 1.000000 | 5.1719 | keep LAB as context only unless a fresh larger forced-flow event appears |
| NEAR | mixed_liquidation_flow_watch | low_liquidation_intensity_context | 3057 | 32948090 | 0.000093 | 0.000019 | -0.322001 | 5.0619 | keep NEAR as context only unless a fresh larger forced-flow event appears |
| BCH | long_liquidation_cascade_watch | low_liquidation_intensity_context | 1098 | 22025412 | 0.000050 | 0.000014 | -0.605924 | 4.7581 | keep BCH as context only unless a fresh larger forced-flow event appears |
| SNDK | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 167 | 11794449 | 0.000014 | 0.000002 | 1.000000 | 4.3685 | keep SNDK as context only unless a fresh larger forced-flow event appears |
| XAG | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 163 | 20718597 | 0.000008 | 0.000001 | 1.000000 | 4.2935 | keep XAG as context only unless a fresh larger forced-flow event appears |
| XRP | long_liquidation_cascade_watch | low_liquidation_intensity_context | 136 | 80223708 | 0.000002 | 0.000001 | -1.000000 | 4.1515 | keep XRP as context only unless a fresh larger forced-flow event appears |

## Interpretation

High liquidation-to-OI rows are more likely to be true forced-flow events than rows that are merely large in dollar terms. The next test is still forward labeling with depth, fees, funding, and adverse-excursion checks.
