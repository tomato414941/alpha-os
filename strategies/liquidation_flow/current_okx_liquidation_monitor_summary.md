# Current OKX Liquidation Monitor

This repeats the OKX liquidation-flow screen over a short window. It is a persistence check, not a trade instruction.

| asset | action | obs | mean score | min score | mean liq USD | mean liq/vol | mean imbalance | latest liquidation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BEAT | short_liquidation_squeeze_watch | 3 | 0.127525 | 0.127223 | 174480 | 0.000592 | 1.000000 | 2026-06-07T15:43:11.119000+00:00 |
| WLD | short_liquidation_squeeze_watch | 3 | 0.117841 | 0.115623 | 522179 | 0.001052 | 0.635519 | 2026-06-07T15:46:29.788000+00:00 |
| BSB | mixed_liquidation_flow_watch | 3 | 0.109096 | 0.108608 | 597731 | 0.002554 | 0.373725 | 2026-06-07T15:45:43.900000+00:00 |
| JTO | long_liquidation_cascade_watch | 3 | 0.094599 | 0.094547 | 41203 | 0.000710 | -0.769187 | 2026-06-07T15:44:30.934000+00:00 |
| ZEC | mixed_liquidation_flow_watch | 3 | 0.050522 | 0.050520 | 403706 | 0.000380 | 0.462134 | 2026-06-07T15:45:59.149000+00:00 |
| LTC | long_liquidation_cascade_watch | 3 | 0.035068 | 0.035064 | 4187 | 0.000094 | -1.000000 | 2026-06-07T15:12:58.783000+00:00 |
| XLM | long_liquidation_cascade_watch | 3 | 0.023186 | 0.023186 | 3313 | 0.000043 | -1.000000 | 2026-06-07T15:00:59.715000+00:00 |
| H | short_liquidation_squeeze_watch | 3 | 0.021281 | 0.021270 | 4007 | 0.000035 | 1.000000 | 2026-06-07T15:44:22.067000+00:00 |
| HOME | mixed_liquidation_flow_watch | 3 | 0.018062 | 0.018056 | 17003 | 0.000305 | -0.244667 | 2026-06-07T15:44:07.932000+00:00 |
| ONDO | short_liquidation_squeeze_watch | 3 | 0.016723 | 0.016715 | 1162 | 0.000030 | 1.000000 | 2026-06-07T15:26:20.101000+00:00 |
| DOGE | long_liquidation_cascade_watch | 3 | 0.015966 | 0.015966 | 5941 | 0.000018 | -1.000000 | 2026-06-07T15:39:43.511000+00:00 |
| NEAR | short_liquidation_squeeze_watch | 3 | 0.010032 | 0.010025 | 1987 | 0.000017 | 0.736140 | 2026-06-07T15:38:46.231000+00:00 |
| ETH | mixed_liquidation_flow_watch | 3 | 0.008588 | 0.007746 | 233466 | 0.000033 | 0.278338 | 2026-06-07T15:41:49.695000+00:00 |
| HYPE | long_liquidation_cascade_watch | 3 | 0.006522 | 0.006521 | 1653 | 0.000004 | -1.000000 | 2026-06-07T14:50:00.881000+00:00 |
| LAB | short_liquidation_squeeze_watch | 3 | 0.006273 | 0.006270 | 1721 | 0.000004 | 0.924577 | 2026-06-07T15:33:18.609000+00:00 |
| BTC | mixed_liquidation_flow_watch | 3 | 0.005563 | 0.001157 | 368932 | 0.000057 | -0.107330 | 2026-06-07T15:41:49.739000+00:00 |
| ALLO | long_liquidation_cascade_watch | 3 | 0.004683 | 0.004682 | 1174 | 0.000004 | -0.793881 | 2026-06-07T15:28:19.380000+00:00 |
| OPN | mixed_liquidation_flow_watch | 3 | 0.000999 | 0.000999 | 1964 | 0.000030 | 0.055743 | 2026-06-07T15:32:36.896000+00:00 |
| EDEN | long_liquidation_cascade_watch | 3 | 0.000245 | 0.000245 | 5 | 0.000000 | -1.000000 | 2026-06-07T15:15:15.085000+00:00 |

## Interpretation

Rows that appear in every sample are persistence candidates. They still need forward labels, fee assumptions, and venue-depth checks.
