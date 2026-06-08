# Current OKX Liquidation Monitor

This repeats the OKX liquidation-flow screen over a short window. It is a persistence check, not a trade instruction.

| asset | action | obs | mean score | min score | mean liq USD | mean liq/vol | mean imbalance | latest liquidation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MU | long_liquidation_cascade_watch | 3 | 0.261991 | 0.261878 | 201391 | 0.002440 | -1.000000 | 2026-06-08T00:00:06.001000+00:00 |
| XAU | short_liquidation_squeeze_watch | 3 | 0.113368 | 0.113362 | 51243 | 0.000579 | 1.000000 | 2026-06-07T23:55:27.333000+00:00 |
| BEAT | long_liquidation_cascade_watch | 3 | 0.069987 | 0.069938 | 238458 | 0.000519 | -0.571470 | 2026-06-08T00:02:05.039000+00:00 |
| HOME | short_liquidation_squeeze_watch | 3 | 0.047642 | 0.047315 | 8840 | 0.000146 | 1.000000 | 2026-06-08T00:00:19.844000+00:00 |
| ETH | short_liquidation_squeeze_watch | 3 | 0.027553 | 0.027551 | 232837 | 0.000026 | 1.000000 | 2026-06-08T00:00:13.744000+00:00 |
| BTC | short_liquidation_squeeze_watch | 3 | 0.024579 | 0.024577 | 197164 | 0.000022 | 1.000000 | 2026-06-07T23:49:10.744000+00:00 |
| ALLO | mixed_liquidation_flow_watch | 3 | 0.022347 | 0.022296 | 30703 | 0.000119 | 0.457491 | 2026-06-08T00:04:12.373000+00:00 |
| WLD | mixed_liquidation_flow_watch | 3 | 0.017785 | 0.017777 | 33973 | 0.000065 | 0.486036 | 2026-06-08T00:01:29.109000+00:00 |
| ZEC | short_liquidation_squeeze_watch | 3 | 0.015398 | 0.015390 | 30427 | 0.000026 | 0.673671 | 2026-06-08T00:05:54.478000+00:00 |
| BSB | short_liquidation_squeeze_watch | 3 | 0.015270 | 0.015262 | 17782 | 0.000050 | 0.507429 | 2026-06-08T00:05:11.337000+00:00 |
| CL | long_liquidation_cascade_watch | 3 | 0.010592 | 0.010591 | 782 | 0.000013 | -1.000000 | 2026-06-07T23:31:27.687000+00:00 |
| LAB | long_liquidation_cascade_watch | 3 | 0.005421 | 0.005420 | 823 | 0.000003 | -1.000000 | 2026-06-07T23:54:07.837000+00:00 |
| OPN | mixed_liquidation_flow_watch | 3 | 0.004453 | 0.004452 | 1083 | 0.000022 | -0.313268 | 2026-06-08T00:00:28.812000+00:00 |
| SUI | mixed_liquidation_flow_watch | 3 | 0.002841 | 0.002841 | 479 | 0.000005 | 0.472369 | 2026-06-07T23:49:00.749000+00:00 |
| PEPE | long_liquidation_cascade_watch | 3 | 0.000060 | 0.000060 | 3 | 0.000000 | -1.000000 | 2026-06-07T23:18:59.830000+00:00 |

## Interpretation

Rows that appear in every sample are persistence candidates. They still need forward labels, fee assumptions, and venue-depth checks.
