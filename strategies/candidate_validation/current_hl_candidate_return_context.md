# Current HL Candidate Return Context

This joins current candidate screens to recent Hyperliquid candle returns. It is context, not a causal alpha test.

| symbol | sources | close | 1h | 4h | 24h | vol24h | action | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| WLD | cross_exchange_funding | 0.49185000 | 0.105355 | 0.171267 | 0.177943 | 121923655.90 | single_source_momentum_context | 34.098874 | candidate has a single source and a recent 4h move |
| MEGA | perp_carry_reversion | 0.04627600 | 0.035118 | 0.040261 | 0.063792 | 24245736.00 | single_source_momentum_context | 20.524871 | candidate has a single source and a recent 4h move |
| STABLE | cross_exchange_funding;perp_carry_reversion | 0.03366900 | -0.003463 | 0.001845 | 0.036256 | 35488188.00 | multi_source_watch | 20.438540 | candidate has multiple sources but no large recent move |
| MON | perp_carry_reversion | 0.02226100 | -0.002822 | -0.023426 | 0.061817 | 111443620.00 | single_source_context | 16.453517 | candidate remains visible but needs stronger labels |
| BABY | perp_carry_reversion | 0.01556600 | 0.011699 | -0.001668 | 0.045259 | 107875950.00 | single_source_context | 16.253271 | candidate remains visible but needs stronger labels |
| SAGA | perp_carry_reversion | 0.01370000 | 0.006613 | 0.002928 | 0.026217 | 13674116.60 | single_source_context | 15.807691 | candidate remains visible but needs stronger labels |
| HEMI | perp_carry_reversion | 0.00559000 | 0.004673 | 0.000358 | 0.025500 | 18646015.00 | single_source_context | 15.485185 | candidate remains visible but needs stronger labels |
| PURR | perp_carry_reversion | 0.08961500 | 0.000156 | 0.002461 | 0.007748 | 9037720.00 | single_source_context | 14.176394 | candidate remains visible but needs stronger labels |
| IP | perp_carry_reversion | 0.31710000 | 0.042886 | 0.028610 | 0.025682 | 1227854.30 | single_source_context | 11.946999 | candidate remains visible but needs stronger labels |
| AERO | attention_market_join;perp_carry_reversion | 0.32953000 | 0.006844 | -0.000879 | 0.040544 | 1217829.00 | multi_source_watch | 11.946201 | candidate has multiple sources but no large recent move |
| ZRO | perp_carry_reversion | 0.89982000 | 0.004387 | -0.007544 | -0.004062 | 2539049.20 | single_source_context | 8.354928 | candidate remains visible but needs stronger labels |
| XMR | perp_carry_reversion | 312.07000000 | 0.014070 | 0.030989 | 0.048129 | 26059.54 | single_source_momentum_context | 7.982531 | candidate has a single source and a recent 4h move |
| SNX | perp_carry_reversion | 0.24161000 | 0.007548 | 0.002656 | 0.035753 | 1025509.60 | single_source_context | 6.913102 | candidate remains visible but needs stronger labels |
| ATOM | perp_carry_reversion | 1.68630000 | 0.010184 | 0.001663 | 0.044343 | 250636.50 | single_source_context | 6.352188 | candidate remains visible but needs stronger labels |
| MORPHO | perp_carry_reversion | 1.69840000 | 0.003961 | -0.009217 | 0.042411 | 472379.80 | single_source_context | 6.329287 | candidate remains visible but needs stronger labels |
| UMA | perp_carry_reversion | 0.38154000 | 0.006410 | -0.003005 | 0.027606 | 192036.40 | single_source_context | 5.983263 | candidate remains visible but needs stronger labels |

## Interpretation

`multi_source_momentum_context` means a candidate appears in more than one research lane and has a recent directional move. `single_source_context` keeps a candidate visible but lower priority. Future-return labels are still needed before this becomes evidence of alpha.
