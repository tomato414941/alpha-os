# BTC ETF Flow Regime Summary

This splits BTC ETF flow forward labels by action and flow-size regimes. It is not net PnL.

| group | obs | mean flow BTC | mean 5d flow BTC | mean dir 1d | mean dir 3d | mean dir 5d | hit 5d | action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| large_5d_outflow | 53 | -5491.98 | -25389.53 | 0.00500591 | 0.02192655 | 0.03437137 | 0.7170 | regime_candidate |
| btc_etf_distribution_label | 89 | -5566.01 | -16890.39 | -0.00270938 | 0.00857785 | 0.01862256 | 0.5843 | regime_candidate |
| btc_etf_inflow_context_label | 139 | -1393.24 | 15883.07 | 0.00414234 | 0.01047907 | 0.01637483 | 0.6187 | regime_candidate |
| large_5d_inflow | 226 | 7099.85 | 36532.69 | 0.00357731 | 0.00609319 | 0.01522096 | 0.5796 | regime_candidate |
| small_daily_flow | 351 | 116.30 | 8027.75 | 0.00368023 | 0.01061866 | 0.01413911 | 0.5954 | regime_candidate |
| large_daily_outflow | 48 | -9175.10 | -13288.79 | -0.00477534 | -0.00004195 | 0.01404040 | 0.5208 | regime_watch |
| btc_etf_accumulation_label | 241 | 8280.52 | 29708.39 | 0.00148820 | 0.00413716 | 0.01089986 | 0.5809 | regime_candidate |
| large_daily_inflow | 156 | 11573.54 | 31705.22 | 0.00148476 | 0.00255431 | 0.00900873 | 0.5385 | regime_watch |
| mixed_5d_flow | 276 | 334.77 | 779.53 | 0.00079848 | 0.00574072 | 0.00645112 | 0.5399 | regime_watch |
| btc_etf_outflow_context_label | 86 | 1154.87 | -8585.24 | 0.00698675 | 0.01054109 | 0.00560175 | 0.4651 | regime_watch |

## Interpretation

The useful question is not whether ETF flow is universally predictive, but which flow regime survives leakage-safe forward labeling.
