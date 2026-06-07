# Spot/Perp Carry Period Audit

This checks whether the 14-day spot/perp carry candidate persists across calendar periods. It uses the broad spot/perp common universe and the default paired-leg cost.

| period | candidate | steps | total return | sharpe | max drawdown | turnover |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | spot_perp_positive_funding_top_3_14d | 365 | 0.067498 | 9.172604 | -0.001419 | 0.119635 |
| 2024 | spot_perp_positive_funding_top_2_14d | 365 | 0.072672 | 8.473524 | -0.001532 | 0.126027 |
| 2024 | spot_perp_positive_funding_top_1_14d | 365 | 0.073596 | 6.525581 | -0.002207 | 0.134247 |
| 2025 | spot_perp_positive_funding_top_3_14d | 364 | -0.000091 | -0.014587 | -0.005839 | 0.119963 |
| 2025 | spot_perp_positive_funding_top_2_14d | 364 | -0.005073 | -0.772013 | -0.009759 | 0.126374 |
| 2025 | spot_perp_positive_funding_top_1_14d | 364 | -0.014489 | -1.462505 | -0.017585 | 0.134615 |
| 2026_to_date | spot_perp_positive_funding_top_1_14d | 150 | -0.005269 | -0.931406 | -0.007119 | 0.126667 |
| 2026_to_date | spot_perp_positive_funding_top_2_14d | 150 | -0.007008 | -1.737634 | -0.009312 | 0.120000 |
| 2026_to_date | spot_perp_positive_funding_top_3_14d | 150 | -0.006695 | -1.744162 | -0.008462 | 0.111111 |

## Interpretation

A deployable carry candidate should not depend on one narrow calendar slice. If the edge is concentrated in one historical period, the next work should treat it as a current dislocation monitor rather than a stable historical strategy.
