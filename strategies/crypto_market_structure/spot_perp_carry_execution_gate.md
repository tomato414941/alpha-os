# Spot/Perp Carry Execution Gate

This compares each carry candidate's fee ceiling with simple execution scenarios. The scenarios are assumptions, not exchange fee schedules.

| candidate | scenario | ceiling bps | scenario bps | headroom bps | pass | default sharpe | turnover |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| spot_perp_positive_funding_top_1_14d | low_slippage_maker_like | 9.918648 | 6.000000 | 3.918648 | True | 2.505911 | 0.121453 |
| spot_perp_positive_funding_top_2_14d | low_slippage_maker_like | 9.589022 | 6.000000 | 3.589022 | True | 2.922381 | 0.119183 |
| spot_perp_positive_funding_top_3_14d | low_slippage_maker_like | 9.451312 | 6.000000 | 3.451312 | True | 3.038431 | 0.115399 |
| spot_perp_positive_funding_top_1_14d | low_slippage_taker_like | 9.918648 | 7.500000 | 2.418648 | True | 2.505911 | 0.121453 |
| spot_perp_positive_funding_top_2_14d | low_slippage_taker_like | 9.589022 | 7.500000 | 2.089022 | True | 2.922381 | 0.119183 |
| spot_perp_positive_funding_top_3_14d | low_slippage_taker_like | 9.451312 | 7.500000 | 1.951312 | True | 3.038431 | 0.115399 |
| spot_perp_positive_funding_top_1_7d | low_slippage_maker_like | 5.324513 | 6.000000 | -0.675487 | False | 1.146710 | 0.255392 |
| spot_perp_positive_funding_top_2_7d | low_slippage_maker_like | 5.309325 | 6.000000 | -0.690675 | False | 1.318138 | 0.239501 |
| spot_perp_positive_funding_top_3_7d | low_slippage_maker_like | 5.194306 | 6.000000 | -0.805694 | False | 1.279093 | 0.236095 |
| spot_perp_positive_funding_top_1_14d | retail_taker_with_slippage | 9.918648 | 11.250000 | -1.331352 | False | 2.505911 | 0.121453 |
| spot_perp_positive_funding_top_2_14d | retail_taker_with_slippage | 9.589022 | 11.250000 | -1.660978 | False | 2.922381 | 0.119183 |
| spot_perp_positive_funding_top_3_14d | retail_taker_with_slippage | 9.451312 | 11.250000 | -1.798688 | False | 3.038431 | 0.115399 |
| spot_perp_positive_funding_top_1_7d | low_slippage_taker_like | 5.324513 | 7.500000 | -2.175487 | False | 1.146710 | 0.255392 |
| spot_perp_positive_funding_top_2_7d | low_slippage_taker_like | 5.309325 | 7.500000 | -2.190675 | False | 1.318138 | 0.239501 |
| spot_perp_positive_funding_top_3_7d | low_slippage_taker_like | 5.194306 | 7.500000 | -2.305694 | False | 1.279093 | 0.236095 |
| spot_perp_positive_funding_top_1_3d | low_slippage_maker_like | 2.708906 | 6.000000 | -3.291094 | False | -2.483362 | 0.568672 |

## Interpretation

A candidate only graduates from historical carry screen to execution watch when it has positive headroom under at least one realistic fee/slippage scenario. The low-turnover 14-day cluster is the only current cluster with meaningful room.
