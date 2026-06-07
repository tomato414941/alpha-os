# Current HL Candidate Return Context

This joins current candidate screens to recent Hyperliquid candle returns. It is context, not a causal alpha test.

| symbol | sources | close | 1h | 4h | 24h | vol24h | action | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| WLD | cross_exchange_funding | 0.48702000 | -0.045714 | 0.144099 | 0.214695 | 143488865.00 | single_source_momentum_context | 26.776315 | candidate has a single source and a recent 4h move |
| STABLE | cross_exchange_funding;perp_carry_reversion | 0.03440400 | 0.020981 | 0.032750 | 0.050568 | 36791971.00 | multi_source_momentum_context | 23.735608 | candidate has multiple sources and a recent 1h move |
| MEGA | perp_carry_reversion | 0.04846500 | 0.032356 | 0.101552 | 0.101027 | 28396102.00 | single_source_momentum_context | 23.313251 | candidate has a single source and a recent 4h move |
| ONDO | l2_imbalance_monitor | 0.34891000 | -0.008412 | 0.042487 | 0.068016 | 42801976.00 | single_source_momentum_context | 17.965574 | candidate has a single source and a recent 4h move |
| XPL | l2_imbalance_monitor | 0.06878500 | -0.015247 | 0.022430 | 0.054225 | 105906498.00 | single_source_context | 17.646195 | candidate remains visible but needs stronger labels |
| XLM | l2_imbalance_monitor | 0.20475000 | 0.008372 | 0.014216 | -0.033286 | 35764464.00 | single_source_context | 16.548051 | candidate remains visible but needs stronger labels |
| BABY | perp_carry_reversion | 0.01563200 | 0.003982 | 0.018239 | 0.024579 | 104070325.00 | single_source_context | 16.310135 | candidate remains visible but needs stronger labels |
| LIT | l2_imbalance_monitor | 1.39920000 | -0.003064 | 0.019825 | -0.047126 | 11228074.00 | single_source_context | 16.297631 | candidate remains visible but needs stronger labels |
| SUI | l2_imbalance_monitor | 0.74630000 | -0.003804 | 0.015982 | 0.043338 | 44146290.20 | single_source_context | 16.179549 | candidate remains visible but needs stronger labels |
| MON | perp_carry_reversion | 0.02215300 | -0.007527 | -0.007793 | 0.055206 | 115427467.00 | single_source_context | 16.142317 | candidate remains visible but needs stronger labels |
| PURR | perp_carry_reversion | 0.08924800 | -0.011508 | 0.020946 | 0.025332 | 8917915.00 | single_source_context | 16.115969 | candidate remains visible but needs stronger labels |
| SAGA | perp_carry_reversion | 0.01365000 | -0.002193 | 0.017139 | 0.016381 | 12901028.10 | single_source_context | 16.076228 | candidate remains visible but needs stronger labels |
| DOGE | l2_imbalance_monitor | 0.08483600 | 0.000613 | 0.017792 | 0.045667 | 110559015.00 | single_source_context | 15.950922 | candidate remains visible but needs stronger labels |
| HEMI | perp_carry_reversion | 0.00557900 | 0.000359 | 0.006132 | 0.011055 | 14991317.00 | single_source_context | 15.342444 | candidate remains visible but needs stronger labels |
| AERO | attention_market_join;perp_carry_reversion | 0.33075000 | -0.000332 | 0.014415 | 0.038364 | 1093933.00 | multi_source_watch | 11.847928 | candidate has multiple sources but no large recent move |
| SOL | l2_imbalance_monitor | 65.15800000 | 0.001291 | 0.020789 | 0.054183 | 4987504.62 | single_source_context | 11.156052 | candidate remains visible but needs stronger labels |
| IP | perp_carry_reversion | 0.31276000 | -0.006701 | 0.029527 | 0.004367 | 1304285.40 | single_source_context | 8.450752 | candidate remains visible but needs stronger labels |
| ZRO | perp_carry_reversion | 0.90516000 | 0.003615 | 0.015641 | -0.004137 | 2298058.30 | single_source_context | 8.441591 | candidate remains visible but needs stronger labels |
| VVV | l2_imbalance_monitor | 16.85500000 | -0.004136 | 0.032149 | 0.050287 | 971492.30 | single_source_momentum_context | 7.992553 | candidate has a single source and a recent 4h move |
| XMR | perp_carry_reversion | 308.36000000 | -0.008074 | 0.021973 | 0.018496 | 26707.65 | single_source_context | 6.932783 | candidate remains visible but needs stronger labels |
| SNX | perp_carry_reversion | 0.24122000 | -0.001490 | 0.013104 | 0.030238 | 929213.90 | single_source_context | 6.733420 | candidate remains visible but needs stronger labels |
| ATOM | perp_carry_reversion | 1.68940000 | -0.002480 | 0.018263 | 0.045033 | 232867.21 | single_source_context | 6.394005 | candidate remains visible but needs stronger labels |
| ETH | l2_imbalance_monitor | 1630.30000000 | -0.001042 | 0.009349 | 0.044261 | 402170.85 | single_source_context | 5.973772 | candidate remains visible but needs stronger labels |
| MORPHO | perp_carry_reversion | 1.69950000 | -0.003576 | -0.000294 | 0.037546 | 501665.20 | single_source_context | 5.874016 | candidate remains visible but needs stronger labels |
| UMA | perp_carry_reversion | 0.38092000 | -0.000420 | 0.013166 | 0.039600 | 165428.60 | single_source_context | 5.865712 | candidate remains visible but needs stronger labels |

## Interpretation

`multi_source_momentum_context` means a candidate appears in more than one research lane and has a recent directional move. `single_source_context` keeps a candidate visible but lower priority. Future-return labels are still needed before this becomes evidence of alpha.
