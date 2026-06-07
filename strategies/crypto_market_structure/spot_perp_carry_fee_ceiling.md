# Spot/Perp Carry Fee Ceiling

This estimates the maximum paired-leg cost before each spot/perp carry candidate loses positive total return. It is based on the same historical spot/perp approximation as `spot_perp_carry.py`.

| candidate | max paired-leg cost bps | zero-cost total | zero-cost sharpe | default total | default sharpe | drawdown | turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| spot_perp_positive_funding_top_3_14d | 12.475663 | 0.089058 | 8.298997 | 0.059689 | 4.967699 | -0.003341 | 0.077563 |
| spot_perp_positive_funding_top_2_14d | 11.377382 | 0.089145 | 7.896105 | 0.056951 | 4.481852 | -0.003772 | 0.085131 |
| spot_perp_positive_funding_top_1_14d | 10.146755 | 0.086829 | 6.272630 | 0.051752 | 3.406656 | -0.004388 | 0.093076 |
| spot_perp_positive_funding_top_3_7d | 7.097393 | 0.090996 | 8.668619 | 0.038750 | 3.095506 | -0.010138 | 0.139236 |
| spot_perp_positive_funding_top_2_7d | 6.439549 | 0.090858 | 8.226793 | 0.033505 | 2.497691 | -0.011141 | 0.153235 |
| spot_perp_positive_funding_top_1_7d | 5.782354 | 0.089379 | 6.746282 | 0.026748 | 1.734848 | -0.013679 | 0.167991 |
| spot_perp_positive_funding_top_3_3d | 3.173292 | 0.099654 | 9.065239 | -0.024451 | -1.689085 | -0.046676 | 0.339765 |
| spot_perp_positive_funding_top_2_3d | 3.024226 | 0.102621 | 8.924331 | -0.031038 | -2.038645 | -0.050494 | 0.366629 |

## Interpretation

Higher ceilings indicate more execution-cost room. A low-turnover 14-day candidate can survive much higher paired-leg costs than daily or 3-day variants. This still omits exchange-specific margin, borrow, liquidation, and order-book availability.
