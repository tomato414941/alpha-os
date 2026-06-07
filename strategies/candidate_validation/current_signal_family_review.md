# Current Signal Family Review

This aggregates short-horizon labels by signal family. It asks which kind of signal is currently showing support, not only which asset is on top.

| family | obs | cov15 | mean15 | hit15 | max15 | min15 | score | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| okx_liquidation:short_liquidation_squeeze_watch | 17 | 17 | 0.003543 | 0.882353 | 0.019775 | -0.001606 | 1.151348 | supported by first labels |
| okx_pressure:long_carry_discount_watch | 32 | 32 | 0.000710 | 0.718750 | 0.007264 | -0.005525 | 0.155283 | supported by first labels |
| hl_candidate:okx_hl_current:paper_24h_monitor | 22 | 22 | 0.008023 | 0.454545 | 0.019682 | -0.001692 | 0.000000 | positive mean but weak hit rate |
| hl_candidate:perp_carry_reversion:long_carry_reversion_watch | 120 | 114 | 0.000497 | 0.280702 | 0.017831 | -0.007837 | 0.000000 | positive mean but weak hit rate |
| hl_candidate:perp_carry_reversion:short_carry_reversion_watch | 30 | 30 | 0.000858 | 0.433333 | 0.011059 | -0.005924 | 0.000000 | positive mean but weak hit rate |
| okx_pressure:short_carry_watch | 45 | 45 | -0.000925 | 0.266667 | 0.002930 | -0.011341 | 0.000000 | not supported by first labels |
| okx_pressure:short_carry_premium_watch | 13 | 13 | -0.003216 | 0.307692 | 0.004037 | -0.030560 | 0.000000 | not supported by first labels |
| okx_pressure:long_carry_watch | 6 | 6 | 0.000663 | 0.500000 | 0.002930 | -0.002685 | 0.000000 | positive mean but weak hit rate |
| okx_pressure:flat_watch | 4 | 0 |  |  |  |  | 0.000000 | waiting for elapsed labels |
| okx_liquidation:long_liquidation_cascade_watch | 8 | 8 | -0.000119 | 0.375000 | 0.005715 | -0.003874 | 0.000000 | not supported by first labels |
| okx_liquidation:mixed_liquidation_flow_watch | 1 | 0 |  |  |  |  | 0.000000 | waiting for elapsed labels |

## Interpretation

This is a small live-label summary. It is useful for prioritizing which signal families deserve repeated sampling, but it is not a backtest or execution-ready PnL estimate.
