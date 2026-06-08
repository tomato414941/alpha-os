# Current Signal Family Review

This aggregates short-horizon labels by signal family. It asks which kind of signal is currently showing support, not only which asset is on top.

| family | obs | cov15 | mean15 | hit15 | max15 | min15 | score | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hl_candidate:perp_carry_reversion:short_carry_reversion_watch | 71 | 0 |  |  |  |  | 0.000000 | waiting for elapsed labels |
| hl_candidate:perp_carry_reversion:long_carry_reversion_watch | 79 | 0 |  |  |  |  | 0.000000 | waiting for elapsed labels |
| hl_candidate:okx_hl_current:paper_24h_monitor | 22 | 0 |  |  |  |  | 0.000000 | waiting for elapsed labels |
| okx_pressure:long_carry_discount_watch | 30 | 0 |  |  |  |  | 0.000000 | waiting for elapsed labels |
| okx_pressure:short_carry_premium_watch | 15 | 0 |  |  |  |  | 0.000000 | waiting for elapsed labels |
| okx_pressure:short_carry_watch | 46 | 0 |  |  |  |  | 0.000000 | waiting for elapsed labels |
| okx_pressure:long_carry_watch | 6 | 0 |  |  |  |  | 0.000000 | waiting for elapsed labels |
| okx_pressure:flat_watch | 3 | 0 |  |  |  |  | 0.000000 | waiting for elapsed labels |
| okx_liquidation:long_liquidation_cascade_watch | 6 | 4 | -0.002422 | 0.250000 | 0.001778 | -0.006230 | 0.000000 | not supported by first labels |
| okx_liquidation:short_liquidation_squeeze_watch | 6 | 3 | -0.003707 | 0.000000 | 0.000000 | -0.008958 | 0.000000 | not supported by first labels |
| okx_liquidation:mixed_liquidation_flow_watch | 4 | 0 |  |  |  |  | 0.000000 | waiting for elapsed labels |
| l2_imbalance:visible_book_imbalance | 22 | 22 | -0.002943 | 0.409091 | 0.007238 | -0.024327 | 0.000000 | not supported by first labels |

## Interpretation

This is a small live-label summary. It is useful for prioritizing which signal families deserve repeated sampling, but it is not a backtest or execution-ready PnL estimate.
