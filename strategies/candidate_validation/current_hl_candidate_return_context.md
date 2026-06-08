# Current HL Candidate Return Context

This joins current candidate screens to recent Hyperliquid candle returns. It is context, not a causal alpha test.

| symbol | sources | close | 1h | 4h | 24h | vol24h | action | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| WLD | cross_exchange_funding;l2_imbalance_monitor | 0.46974000 | -0.015117 | -0.014063 | 0.040399 | 160001030.30 | multi_source_momentum_context | 22.214820 | candidate has multiple sources and a recent 1h move |
| HMSTR | perp_carry_reversion | 0.00017800 | -0.011111 | -0.058201 | -0.016575 | 304650815.00 | single_source_momentum_context | 19.021164 | candidate has a single source and a recent 4h move |
| JTO | l2_imbalance_monitor | 0.58973000 | -0.018523 | -0.039246 | 0.087842 | 15845323.00 | single_source_momentum_context | 18.814647 | candidate has a single source and a recent 4h move |
| AZTEC | perp_carry_reversion | 0.01545900 | -0.011194 | -0.037422 | -0.031330 | 11534942.00 | single_source_momentum_context | 17.990464 | candidate has a single source and a recent 4h move |
| STBL | perp_carry_reversion | 0.02447000 | -0.012032 | -0.032615 | 0.043363 | 14032304.00 | single_source_momentum_context | 17.833922 | candidate has a single source and a recent 4h move |
| ENA | l2_imbalance_monitor | 0.08606100 | -0.008674 | -0.032946 | -0.076341 | 86213761.00 | single_source_momentum_context | 17.514692 | candidate has a single source and a recent 4h move |
| ADA | l2_imbalance_monitor | 0.16016000 | -0.007498 | -0.015006 | -0.008972 | 73743558.00 | single_source_context | 16.500137 | candidate remains visible but needs stronger labels |
| CFX | perp_carry_reversion | 0.04520000 | -0.007858 | -0.012626 | 0.007063 | 19802232.00 | single_source_context | 16.417119 | candidate remains visible but needs stronger labels |
| STABLE | cross_exchange_funding | 0.03287200 | 0.005814 | -0.010416 | -0.012141 | 23709674.00 | single_source_context | 16.102162 | candidate remains visible but needs stronger labels |
| PENGU | attention_market_join | 0.00667500 | -0.001197 | -0.018094 | -0.017371 | 487102198.00 | single_source_context | 16.024385 | candidate remains visible but needs stronger labels |
| HEMI | perp_carry_reversion | 0.00546400 | -0.001097 | -0.015318 | -0.017620 | 13591805.00 | single_source_context | 15.875593 | candidate remains visible but needs stronger labels |
| PURR | perp_carry_reversion | 0.09341600 | 0.006660 | 0.010613 | 0.031173 | 8436261.00 | single_source_context | 14.632867 | candidate remains visible but needs stronger labels |
| DYDX | perp_carry_reversion | 0.14210000 | 0.004453 | -0.004274 | 0.026882 | 6238563.90 | single_source_context | 11.897608 | candidate remains visible but needs stronger labels |
| XMR | l2_imbalance_monitor;perp_carry_reversion | 301.57000000 | -0.014058 | -0.003799 | 0.013749 | 28156.22 | multi_source_momentum_context | 11.623927 | candidate has multiple sources and a recent 1h move |
| ZRO | perp_carry_reversion | 0.88881000 | -0.011742 | -0.013737 | -0.040576 | 3290484.10 | single_source_context | 10.151509 | candidate remains visible but needs stronger labels |
| VIRTUAL | perp_carry_reversion | 0.56648000 | -0.005268 | -0.016920 | -0.007481 | 3224773.30 | single_source_context | 9.597586 | candidate remains visible but needs stronger labels |
| AVAX | l2_imbalance_monitor | 6.55450000 | -0.005432 | -0.019903 | -0.034882 | 1804171.54 | single_source_context | 8.342519 | candidate remains visible but needs stronger labels |
| APEX | perp_carry_reversion | 0.25574000 | -0.011824 | -0.015173 | 0.018884 | 334735.00 | single_source_context | 7.275741 | candidate remains visible but needs stronger labels |
| SNX | perp_carry_reversion | 0.24163000 | -0.005965 | -0.013393 | 0.002281 | 900296.00 | single_source_context | 7.166441 | candidate remains visible but needs stronger labels |
| TRUMP | perp_carry_reversion | 1.62730000 | -0.006411 | -0.013279 | 0.007117 | 713200.90 | single_source_context | 7.018263 | candidate remains visible but needs stronger labels |
| ZEC | attention_market_join | 426.90000000 | 0.003314 | -0.004431 | 0.073800 | 1233690.74 | single_source_context | 6.786622 | candidate remains visible but needs stronger labels |
| BNB | l2_imbalance_monitor | 591.90000000 | -0.008543 | -0.017724 | 0.009586 | 17473.46 | single_source_context | 6.757934 | candidate remains visible but needs stronger labels |
| AAVE | perp_carry_reversion | 61.96000000 | -0.006558 | -0.015070 | -0.012291 | 89996.36 | single_source_context | 6.499252 | candidate remains visible but needs stronger labels |
| ATOM | perp_carry_reversion | 1.69950000 | -0.008055 | -0.003985 | 0.004789 | 206724.54 | single_source_context | 6.211449 | candidate remains visible but needs stronger labels |
| LTC | l2_imbalance_monitor | 42.13900000 | 0.000261 | -0.004159 | 0.009197 | 104177.50 | single_source_context | 5.338252 | candidate remains visible but needs stronger labels |

## Interpretation

`multi_source_momentum_context` means a candidate appears in more than one research lane and has a recent directional move. `single_source_context` keeps a candidate visible but lower priority. Future-return labels are still needed before this becomes evidence of alpha.
