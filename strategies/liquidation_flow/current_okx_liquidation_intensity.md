# Current OKX Liquidation Intensity

This joins recent OKX liquidation notional to OKX open interest. It is an event-intensity screen, not a trade instruction.

| asset | action | status | liq USD | OI USD | liq/OI | liq/vol | imbalance | score | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 327197 | 3916175 | 0.083550 | 0.004769 | -0.351183 | 841.8515 | label H mixed_liquidation_flow_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| PIPPIN | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 98632 | 5807069 | 0.016985 | 0.000505 | -0.704132 | 176.3304 | label PIPPIN long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| WLD | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 204194 | 36377075 | 0.005613 | 0.000412 | -0.615371 | 62.7397 | label WLD long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| SUI | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 80911 | 19625811 | 0.004123 | 0.001221 | -1.000000 | 48.3062 | label SUI long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| FIL | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 46251 | 11219497 | 0.004122 | 0.000906 | -1.000000 | 48.0291 | label FIL long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| SOL | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 574464 | 181785475 | 0.003160 | 0.000828 | -1.000000 | 39.5263 | label SOL long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| DOGE | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 208127 | 67736209 | 0.003073 | 0.000707 | -1.000000 | 38.1857 | label DOGE long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ALLO | short_liquidation_squeeze_watch | liquidation_oi_pressure_watch | 26423 | 8515531 | 0.003103 | 0.000054 | 0.725393 | 36.9253 | label ALLO short_liquidation_squeeze_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ZEC | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 179937 | 60472164 | 0.002976 | 0.000248 | -0.806817 | 36.6909 | label ZEC long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ETH | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 2916137 | 1154358394 | 0.002526 | 0.000363 | -0.965829 | 33.7775 | label ETH long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BTC | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 3558949 | 1792399125 | 0.001986 | 0.000478 | -1.000000 | 28.5504 | label BTC long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| ADA | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 42134 | 19636904 | 0.002146 | 0.000680 | -1.000000 | 28.2021 | label ADA long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BSB | long_liquidation_cascade_watch | forced_flow_oi_shock_watch | 12091 | 5516212 | 0.002192 | 0.000081 | -1.000000 | 28.0384 | label BSB long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BCH | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 40809 | 22301225 | 0.001830 | 0.000425 | -1.000000 | 25.0050 | label BCH long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| PEPE | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 27452 | 16741415 | 0.001640 | 0.000292 | -1.000000 | 22.9118 | label PEPE long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| BNB | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 46037 | 43661746 | 0.001054 | 0.000718 | -1.000000 | 17.3321 | label BNB long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| CL | mixed_liquidation_flow_watch | liquidation_oi_pressure_watch | 29263 | 25569953 | 0.001144 | 0.000194 | -0.069516 | 16.0538 | label CL mixed_liquidation_flow_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| SNDK | long_liquidation_cascade_watch | low_liquidation_intensity_context | 6915 | 9449460 | 0.000732 | 0.000066 | -1.000000 | 13.1890 | keep SNDK as context only unless a fresh larger forced-flow event appears |
| HYPE | long_liquidation_cascade_watch | liquidation_oi_pressure_watch | 52785 | 91533172 | 0.000577 | 0.000096 | -0.935728 | 12.4040 | label HYPE long_liquidation_cascade_watch over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion |
| XRP | long_liquidation_cascade_watch | low_liquidation_intensity_context | 27969 | 81173357 | 0.000345 | 0.000160 | -1.000000 | 9.9485 | keep XRP as context only unless a fresh larger forced-flow event appears |
| NEAR | long_liquidation_cascade_watch | low_liquidation_intensity_context | 10104 | 29920370 | 0.000338 | 0.000072 | -1.000000 | 9.4156 | keep NEAR as context only unless a fresh larger forced-flow event appears |
| BEAT | short_liquidation_squeeze_watch | low_liquidation_intensity_context | 13234 | 41569936 | 0.000318 | 0.000019 | 0.662887 | 8.6431 | keep BEAT as context only unless a fresh larger forced-flow event appears |
| LAB | long_liquidation_cascade_watch | low_liquidation_intensity_context | 4273 | 15860735 | 0.000269 | 0.000045 | -0.874135 | 8.0945 | keep LAB as context only unless a fresh larger forced-flow event appears |
| MU | long_liquidation_cascade_watch | low_liquidation_intensity_context | 4004 | 24598770 | 0.000163 | 0.000014 | -1.000000 | 7.2439 | keep MU as context only unless a fresh larger forced-flow event appears |
| XAU | long_liquidation_cascade_watch | low_liquidation_intensity_context | 2541 | 101347936 | 0.000025 | 0.000011 | -1.000000 | 5.6671 | keep XAU as context only unless a fresh larger forced-flow event appears |

## Interpretation

High liquidation-to-OI rows are more likely to be true forced-flow events than rows that are merely large in dollar terms. The next test is still forward labeling with depth, fees, funding, and adverse-excursion checks.
