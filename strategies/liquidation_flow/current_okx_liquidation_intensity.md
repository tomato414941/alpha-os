# Current OKX Liquidation Intensity

This joins recent OKX liquidation notional to OKX open interest. It is an event-intensity screen, not a trade instruction.

| asset | action | status | liq USD | OI USD | liq/OI | liq/vol | imbalance | score | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BEAT | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 238458 | 38398556 | 0.006210 | 0.000517 | -0.571470 | 68.6909 | label BEAT long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| MU | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 201391 | 34575042 | 0.005825 | 0.002439 | -1.000000 | 65.8134 | label MU long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ALLO | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 30703 | 8460870 | 0.003629 | 0.000119 | 0.457491 | 41.7130 | label ALLO mixed_liquidation_flow_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| HOME | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 9107 | 3392547 | 0.002685 | 0.000150 | 1.000000 | 32.8533 | keep HOME as context only unless a fresh larger forced-flow event appears |
| BSB | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 17782 | 6929225 | 0.002566 | 0.000050 | 0.507429 | 30.9431 | label BSB short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| WLD | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 33973 | 34280113 | 0.000991 | 0.000065 | 0.486036 | 15.4315 | label WLD mixed_liquidation_flow_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| XAU | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 51243 | 102780478 | 0.000499 | 0.000580 | 1.000000 | 11.8087 | keep XAU as context only unless a fresh larger forced-flow event appears |
| ZEC | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 30413 | 61070474 | 0.000498 | 0.000026 | 0.674462 | 10.8274 | keep ZEC as context only unless a fresh larger forced-flow event appears |
| ETH | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 232837 | 1166743266 | 0.000200 | 0.000026 | 1.000000 | 9.3902 | keep ETH as context only unless a fresh larger forced-flow event appears |
| BTC | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 197164 | 1664788511 | 0.000118 | 0.000022 | 1.000000 | 8.5037 | keep BTC as context only unless a fresh larger forced-flow event appears |
| OPN | mixed_liquidation_flow_watch | low_liquidation_intensity_context | 1083 | 2986947 | 0.000363 | 0.000022 | -0.313268 | 7.2909 | keep OPN as context only unless a fresh larger forced-flow event appears |
| LAB | long_liquidation_cascade_watch | low_liquidation_intensity_context | 823 | 18474646 | 0.000045 | 0.000003 | -1.000000 | 5.3659 | keep LAB as context only unless a fresh larger forced-flow event appears |
| CL | long_liquidation_cascade_watch | low_liquidation_intensity_context | 782 | 25496528 | 0.000031 | 0.000013 | -1.000000 | 5.2101 | keep CL as context only unless a fresh larger forced-flow event appears |
| SUI | mixed_liquidation_flow_watch | low_liquidation_intensity_context | 479 | 20262636 | 0.000024 | 0.000005 | 0.472369 | 3.8647 | keep SUI as context only unless a fresh larger forced-flow event appears |
| HYPE | long_liquidation_cascade_watch | low_liquidation_intensity_context | 23 | 91128166 | 0.000000 | 0.000000 | -1.000000 | 3.3700 | keep HYPE as context only unless a fresh larger forced-flow event appears |
| PEPE | long_liquidation_cascade_watch | low_liquidation_intensity_context | 3 | 16749311 | 0.000000 | 0.000000 | -1.000000 | 2.4445 | keep PEPE as context only unless a fresh larger forced-flow event appears |

## Interpretation

High liquidation-to-OI rows are more likely to be true forced-flow events than rows that are merely large in dollar terms. The next test is still forward labeling with depth, fees, funding, and adverse-excursion checks.
