# Current OKX Liquidation Intensity

This joins recent OKX liquidation notional to OKX open interest. It is an event-intensity screen, not a trade instruction.

| asset | action | status | liq USD | OI USD | liq/OI | liq/vol | imbalance | score | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 115189 | 4070689 | 0.028297 | 0.001885 | -0.529284 | 289.2080 | label H long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| WLD | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 136631 | 39326762 | 0.003474 | 0.000274 | -0.315874 | 40.5367 | label WLD mixed_liquidation_flow_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| LAB | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 26830 | 16135627 | 0.001663 | 0.000269 | -1.000000 | 23.1292 | label LAB long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| SPCX | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 32489 | 27788047 | 0.001169 | 0.000413 | -1.000000 | 18.2951 | label SPCX long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| PIPPIN | mixed_liquidation_flow_watch | low_liquidation_intensity_context | 7132 | 5854454 | 0.001218 | 0.000036 | 0.456657 | 16.9598 | keep PIPPIN as context only unless a fresh larger forced-flow event appears |
| BEAT | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 35848 | 40296913 | 0.000890 | 0.000052 | -0.865761 | 15.2104 | label BEAT long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BSB | mixed_liquidation_flow_watch | low_liquidation_intensity_context | 6116 | 5892791 | 0.001038 | 0.000037 | -0.303700 | 14.7793 | keep BSB as context only unless a fresh larger forced-flow event appears |
| ZEC | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 40448 | 59416816 | 0.000681 | 0.000054 | 0.135375 | 11.6897 | label ZEC mixed_liquidation_flow_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ETH | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 367194 | 1179043717 | 0.000311 | 0.000046 | 0.521184 | 9.7412 | keep ETH as context only unless a fresh larger forced-flow event appears |
| PEPE | long_liquidation_cascade_watch | low_liquidation_intensity_context | 5899 | 17785582 | 0.000332 | 0.000063 | -1.000000 | 9.1173 | keep PEPE as context only unless a fresh larger forced-flow event appears |
| HYPE | long_liquidation_cascade_watch | low_liquidation_intensity_context | 16795 | 94186055 | 0.000178 | 0.000028 | -0.966615 | 7.9633 | keep HYPE as context only unless a fresh larger forced-flow event appears |
| BTC | long_liquidation_cascade_watch | low_liquidation_intensity_context | 188997 | 1792577811 | 0.000105 | 0.000025 | -0.718567 | 7.7871 | keep BTC as context only unless a fresh larger forced-flow event appears |
| MRVL | long_liquidation_cascade_watch | low_liquidation_intensity_context | 961 | 10833523 | 0.000089 | 0.000009 | -1.000000 | 5.8793 | keep MRVL as context only unless a fresh larger forced-flow event appears |
| NEAR | long_liquidation_cascade_watch | low_liquidation_intensity_context | 979 | 30853249 | 0.000032 | 0.000007 | -1.000000 | 5.3162 | keep NEAR as context only unless a fresh larger forced-flow event appears |
| ADA | long_liquidation_cascade_watch | low_liquidation_intensity_context | 314 | 20260126 | 0.000015 | 0.000005 | -1.000000 | 4.6575 | keep ADA as context only unless a fresh larger forced-flow event appears |
| SOL | long_liquidation_cascade_watch | low_liquidation_intensity_context | 234 | 186696779 | 0.000001 | 0.000000 | -1.000000 | 4.3840 | keep SOL as context only unless a fresh larger forced-flow event appears |
| FIL | long_liquidation_cascade_watch | low_liquidation_intensity_context | 141 | 11313038 | 0.000012 | 0.000003 | -1.000000 | 4.2781 | keep FIL as context only unless a fresh larger forced-flow event appears |

## Interpretation

High liquidation-to-OI rows are more likely to be true forced-flow events than rows that are merely large in dollar terms. The next test is still forward labeling with depth, fees, funding, and adverse-excursion checks.
