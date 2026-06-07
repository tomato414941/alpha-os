# Current HL Candidate Return Context

This joins current candidate screens to recent Hyperliquid candle returns. It is context, not a causal alpha test.

| symbol | sources | close | 1h | 4h | 24h | vol24h | action | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| MEGA | perp_carry_reversion | 0.05009800 | 0.051728 | 0.132056 | 0.148115 | 30330096.00 | single_source_momentum_context | 26.775569 | candidate has a single source and a recent 4h move |
| ONDO | l2_imbalance_monitor;sector_rotation | 0.34636000 | -0.005770 | 0.027165 | 0.053567 | 42640728.00 | multi_source_watch | 21.935217 | candidate has multiple sources but no large recent move |
| STABLE | cross_exchange_funding;perp_carry_reversion | 0.03408100 | -0.005979 | 0.016281 | 0.047003 | 35589034.00 | multi_source_watch | 21.411987 | candidate has multiple sources but no large recent move |
| XPL | l2_imbalance_monitor;sector_rotation | 0.06888800 | 0.003730 | 0.008299 | 0.058187 | 105753973.00 | multi_source_watch | 20.787957 | candidate has multiple sources but no large recent move |
| WLD | cross_exchange_funding | 0.48503000 | -0.001421 | 0.095618 | 0.204984 | 145067881.20 | single_source_momentum_context | 19.922947 | candidate has a single source and a recent 4h move |
| XLM | l2_imbalance_monitor | 0.20810000 | 0.032140 | 0.025376 | -0.021903 | 35361984.00 | single_source_context | 19.482752 | candidate remains visible but needs stronger labels |
| BABY | perp_carry_reversion | 0.01589700 | 0.022776 | 0.026408 | 0.031938 | 104663601.00 | single_source_context | 18.597930 | candidate remains visible but needs stronger labels |
| FARTCOIN | sector_rotation | 0.11285000 | 0.007589 | 0.020067 | 0.071700 | 43444823.60 | single_source_context | 16.762273 | candidate remains visible but needs stronger labels |
| MON | perp_carry_reversion | 0.02199300 | -0.008431 | -0.016017 | 0.047685 | 130856273.00 | single_source_context | 16.643961 | candidate remains visible but needs stronger labels |
| CC | sector_rotation | 0.16650000 | -0.002875 | -0.020646 | 0.052000 | 20864629.00 | single_source_context | 16.319752 | candidate remains visible but needs stronger labels |
| PUMP | sector_rotation | 0.00150700 | -0.002647 | 0.016869 | 0.077198 | 2213776252.00 | single_source_context | 16.108180 | candidate remains visible but needs stronger labels |
| DOGE | l2_imbalance_monitor | 0.08497200 | 0.003602 | 0.014082 | 0.048933 | 110257146.00 | single_source_context | 16.064359 | candidate remains visible but needs stronger labels |
| LIT | l2_imbalance_monitor | 1.37790000 | -0.003832 | -0.012329 | -0.053770 | 11336752.00 | single_source_context | 15.999613 | candidate remains visible but needs stronger labels |
| TURBO | sector_rotation | 0.00086500 | 0.004646 | 0.010514 | 0.035928 | 81599339.00 | single_source_context | 15.990277 | candidate remains visible but needs stronger labels |
| SUI | l2_imbalance_monitor | 0.74846000 | 0.005197 | 0.008271 | 0.051459 | 44060414.90 | single_source_context | 15.933317 | candidate remains visible but needs stronger labels |
| SAGA | perp_carry_reversion | 0.01364000 | 0.002204 | 0.006642 | 0.016393 | 12867600.70 | single_source_context | 15.552529 | candidate remains visible but needs stronger labels |
| HEMI | perp_carry_reversion | 0.00557300 | 0.002699 | -0.001075 | 0.018830 | 14767534.00 | single_source_context | 15.323654 | candidate remains visible but needs stronger labels |
| PYTH | sector_rotation | 0.03156800 | 0.000888 | 0.002254 | 0.006504 | 16190201.00 | single_source_context | 15.201485 | candidate remains visible but needs stronger labels |
| PURR | perp_carry_reversion | 0.08960400 | -0.000346 | 0.012349 | 0.028808 | 8979676.00 | single_source_context | 14.631698 | candidate remains visible but needs stronger labels |
| POL | sector_rotation | 0.07926700 | 0.004944 | 0.006105 | 0.041384 | 8011545.00 | single_source_context | 13.811243 | candidate remains visible but needs stronger labels |
| XMR | perp_carry_reversion;sector_rotation | 314.40000000 | 0.017081 | 0.039545 | 0.039889 | 27221.48 | multi_source_momentum_context | 13.712548 | candidate has multiple sources and a recent 1h move |
| JUP | sector_rotation | 0.15505000 | 0.003885 | 0.007014 | 0.013863 | 6913273.00 | single_source_context | 12.652466 | candidate remains visible but needs stronger labels |
| AERO | attention_market_join;perp_carry_reversion | 0.33085000 | 0.001180 | 0.013261 | 0.040016 | 1086594.00 | multi_source_watch | 11.867664 | candidate has multiple sources but no large recent move |
| SOL | l2_imbalance_monitor | 65.26400000 | 0.004494 | 0.015466 | 0.054295 | 4930852.62 | single_source_context | 11.153577 | candidate remains visible but needs stronger labels |
| ZEC | sector_rotation | 418.06000000 | -0.010275 | 0.031508 | 0.164772 | 980515.54 | single_source_momentum_context | 8.583393 | candidate has a single source and a recent 4h move |

## Interpretation

`multi_source_momentum_context` means a candidate appears in more than one research lane and has a recent directional move. `single_source_context` keeps a candidate visible but lower priority. Future-return labels are still needed before this becomes evidence of alpha.
