# Current Signal Family Review

This aggregates short-horizon labels by signal family. It asks which kind of signal is currently showing support, not only which asset is on top.

| family | obs | cov15 | mean15 | hit15 | max15 | min15 | score | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| okx_liquidation:short_liquidation_squeeze_watch | 17 | 17 | 0.004270 | 0.882353 | 0.027309 | -0.009277 | 1.387726 | supported by first labels |
| okx_pressure:long_carry_discount_watch | 32 | 32 | 0.001608 | 0.750000 | 0.024699 | -0.008581 | 0.401907 | supported by first labels |
| hl_candidate:okx_hl_current:paper_24h_monitor | 22 | 22 | 0.008023 | 0.454545 | 0.019682 | -0.001692 | 0.000000 | positive mean but weak hit rate |
| hl_candidate:perp_carry_reversion:long_carry_reversion_watch | 120 | 120 | 0.000472 | 0.266667 | 0.017831 | -0.007837 | 0.000000 | positive mean but weak hit rate |
| hl_candidate:perp_carry_reversion:short_carry_reversion_watch | 30 | 30 | 0.000858 | 0.433333 | 0.011059 | -0.005924 | 0.000000 | positive mean but weak hit rate |
| okx_pressure:short_carry_watch | 45 | 45 | -0.001622 | 0.200000 | 0.008225 | -0.010693 | 0.000000 | not supported by first labels |
| okx_pressure:short_carry_premium_watch | 13 | 13 | -0.002705 | 0.230769 | 0.009967 | -0.013383 | 0.000000 | not supported by first labels |
| okx_pressure:long_carry_watch | 6 | 6 | -0.000905 | 0.500000 | 0.004828 | -0.010254 | 0.000000 | not supported by first labels |
| okx_pressure:flat_watch | 4 | 0 |  |  |  |  | 0.000000 | waiting for elapsed labels |
| okx_liquidation:long_liquidation_cascade_watch | 8 | 8 | -0.001233 | 0.375000 | 0.002599 | -0.007351 | 0.000000 | not supported by first labels |
| okx_liquidation:mixed_liquidation_flow_watch | 1 | 0 |  |  |  |  | 0.000000 | waiting for elapsed labels |

## Interpretation

This is a small live-label summary. It is useful for prioritizing which signal families deserve repeated sampling, but it is not a backtest or execution-ready PnL estimate.
