# Spot/Perp Carry Fee Ceiling

This estimates the maximum paired-leg cost before each spot/perp carry candidate loses positive total return. It is based on the same historical spot/perp approximation as `spot_perp_carry.py`.

| candidate | max paired-leg cost bps | zero-cost total | zero-cost sharpe | default total | default sharpe | drawdown | turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| spot_perp_positive_funding_top_1_14d | 9.918648 | 0.112061 | 4.439976 | 0.065458 | 2.505911 | -0.012568 | 0.121453 |
| spot_perp_positive_funding_top_2_14d | 9.589022 | 0.106008 | 5.419287 | 0.060508 | 2.922381 | -0.016324 | 0.119183 |
| spot_perp_positive_funding_top_3_14d | 9.451312 | 0.100927 | 5.746814 | 0.057046 | 3.038431 | -0.015078 | 0.115399 |
| spot_perp_positive_funding_top_1_7d | 5.324513 | 0.127317 | 5.068643 | 0.030272 | 1.146710 | -0.029210 | 0.255392 |
| spot_perp_positive_funding_top_2_7d | 5.309325 | 0.118583 | 6.170666 | 0.028031 | 1.318138 | -0.028262 | 0.239501 |
| spot_perp_positive_funding_top_3_7d | 5.194306 | 0.114128 | 6.476165 | 0.025169 | 1.279093 | -0.027635 | 0.236095 |
| spot_perp_positive_funding_top_1_3d | 2.708906 | 0.145363 | 6.212902 | -0.062662 | -2.483362 | -0.095473 | 0.568672 |
| spot_perp_positive_funding_top_2_3d | 2.520025 | 0.128019 | 6.688512 | -0.068325 | -3.097790 | -0.095937 | 0.542565 |

## Interpretation

Higher ceilings indicate more execution-cost room. A low-turnover 14-day candidate can survive much higher paired-leg costs than daily or 3-day variants. This still omits exchange-specific margin, borrow, liquidation, and order-book availability.
