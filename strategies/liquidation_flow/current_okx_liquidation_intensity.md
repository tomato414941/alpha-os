# Current OKX Liquidation Intensity

This joins recent OKX liquidation notional to OKX open interest. It is an event-intensity screen, not a trade instruction.

| asset | action | status | liq USD | OI USD | liq/OI | liq/vol | imbalance | score | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| XAU | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 1252616 | 102710148 | 0.012196 | 0.007966 | -0.996520 | 130.5896 | label XAU long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| XAG | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 226105 | 22793136 | 0.009920 | 0.003298 | -0.941572 | 106.7256 | label XAG long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BSB | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 47760 | 7081174 | 0.006745 | 0.000141 | 0.941290 | 74.0609 | label BSB short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BEAT | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 177381 | 38254524 | 0.004637 | 0.000254 | 1.000000 | 53.7012 | label BEAT short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ALLO | short_liquidation_squeeze_watch | forced_flow_oi_shock_watch | 34946 | 8520167 | 0.004102 | 0.000138 | 0.775715 | 47.1519 | label ALLO short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| PIPPIN | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 15631 | 4394661 | 0.003557 | 0.000242 | -0.658067 | 41.1213 | label PIPPIN long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| WLD | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 81723 | 34583986 | 0.002363 | 0.000179 | -0.652272 | 29.8900 | label WLD long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| HOME | mixed_liquidation_flow_watch | low_liquidation_intensity_context | 7355 | 3399503 | 0.002164 | 0.000091 | 0.383643 | 26.2848 | keep HOME as context only unless a fresh larger forced-flow event appears |
| SUI | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 31704 | 20238385 | 0.001567 | 0.000366 | -1.000000 | 22.2527 | label SUI long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| MU | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 37995 | 34552239 | 0.001100 | 0.000253 | -1.000000 | 17.6489 | label MU long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| DOGE | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 58987 | 68890237 | 0.000856 | 0.000162 | -1.000000 | 15.3939 | label DOGE long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| HYPE | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 76460 | 91484242 | 0.000836 | 0.000161 | 1.000000 | 15.3030 | label HYPE short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ETH | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 840956 | 1164122115 | 0.000722 | 0.000092 | 0.475510 | 14.1268 | label ETH mixed_liquidation_flow_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| LAB | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 13391 | 18610404 | 0.000720 | 0.000073 | 1.000000 | 13.3575 | label LAB short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| PEPE | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 10493 | 16709498 | 0.000628 | 0.000076 | -1.000000 | 12.3356 | label PEPE long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BTC | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 657618 | 1663315105 | 0.000395 | 0.000070 | 0.593877 | 10.9883 | keep BTC as context only unless a fresh larger forced-flow event appears |
| MRVL | long_liquidation_cascade_watch | low_liquidation_intensity_context | 4872 | 10455741 | 0.000466 | 0.000096 | -1.000000 | 10.3834 | keep MRVL as context only unless a fresh larger forced-flow event appears |
| NEAR | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 12531 | 30531837 | 0.000410 | 0.000085 | 0.946882 | 10.1316 | keep NEAR as context only unless a fresh larger forced-flow event appears |
| SOL | long_liquidation_cascade_watch | low_liquidation_intensity_context | 41505 | 181759285 | 0.000228 | 0.000052 | -1.000000 | 8.9347 | keep SOL as context only unless a fresh larger forced-flow event appears |
| ZEC | mixed_liquidation_flow_watch | low_liquidation_intensity_context | 19597 | 61129846 | 0.000321 | 0.000020 | 0.336683 | 8.1777 | keep ZEC as context only unless a fresh larger forced-flow event appears |
| XRP | long_liquidation_cascade_watch | low_liquidation_intensity_context | 14661 | 79725472 | 0.000184 | 0.000060 | -1.000000 | 8.0373 | keep XRP as context only unless a fresh larger forced-flow event appears |
| JTO | long_liquidation_cascade_watch | low_liquidation_intensity_context | 1000 | 3559057 | 0.000281 | 0.000015 | -0.944093 | 7.7085 | keep JTO as context only unless a fresh larger forced-flow event appears |
| SNDK | long_liquidation_cascade_watch | low_liquidation_intensity_context | 758 | 11559651 | 0.000066 | 0.000014 | -1.000000 | 5.5467 | keep SNDK as context only unless a fresh larger forced-flow event appears |
| TON | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 781 | 15829845 | 0.000049 | 0.000014 | 0.568555 | 4.5288 | keep TON as context only unless a fresh larger forced-flow event appears |
| CL | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 235 | 25496503 | 0.000009 | 0.000002 | 1.000000 | 4.4656 | keep CL as context only unless a fresh larger forced-flow event appears |

## Interpretation

High liquidation-to-OI rows are more likely to be true forced-flow events than rows that are merely large in dollar terms. The next test is still forward labeling with depth, fees, funding, and adverse-excursion checks.
