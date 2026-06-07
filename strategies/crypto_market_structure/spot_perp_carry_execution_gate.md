# Spot/Perp Carry Execution Gate

This compares each carry candidate's fee ceiling with simple execution scenarios. The scenarios are assumptions, not exchange fee schedules.

| candidate | scenario | ceiling bps | scenario bps | headroom bps | pass | default sharpe | turnover |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| spot_perp_positive_funding_top_3_14d | low_slippage_maker_like | 12.475663 | 6.000000 | 6.475663 | True | 4.967699 | 0.077563 |
| spot_perp_positive_funding_top_2_14d | low_slippage_maker_like | 11.377382 | 6.000000 | 5.377382 | True | 4.481852 | 0.085131 |
| spot_perp_positive_funding_top_3_14d | low_slippage_taker_like | 12.475663 | 7.500000 | 4.975663 | True | 4.967699 | 0.077563 |
| spot_perp_positive_funding_top_1_14d | low_slippage_maker_like | 10.146755 | 6.000000 | 4.146755 | True | 3.406656 | 0.093076 |
| spot_perp_positive_funding_top_2_14d | low_slippage_taker_like | 11.377382 | 7.500000 | 3.877382 | True | 4.481852 | 0.085131 |
| spot_perp_positive_funding_top_1_14d | low_slippage_taker_like | 10.146755 | 7.500000 | 2.646755 | True | 3.406656 | 0.093076 |
| spot_perp_positive_funding_top_3_14d | retail_taker_with_slippage | 12.475663 | 11.250000 | 1.225663 | True | 4.967699 | 0.077563 |
| spot_perp_positive_funding_top_3_7d | low_slippage_maker_like | 7.097393 | 6.000000 | 1.097393 | True | 3.095506 | 0.139236 |
| spot_perp_positive_funding_top_2_7d | low_slippage_maker_like | 6.439549 | 6.000000 | 0.439549 | True | 2.497691 | 0.153235 |
| spot_perp_positive_funding_top_2_14d | retail_taker_with_slippage | 11.377382 | 11.250000 | 0.127382 | True | 4.481852 | 0.085131 |
| spot_perp_positive_funding_top_1_7d | low_slippage_maker_like | 5.782354 | 6.000000 | -0.217646 | False | 1.734848 | 0.167991 |
| spot_perp_positive_funding_top_3_7d | low_slippage_taker_like | 7.097393 | 7.500000 | -0.402607 | False | 3.095506 | 0.139236 |
| spot_perp_positive_funding_top_2_7d | low_slippage_taker_like | 6.439549 | 7.500000 | -1.060451 | False | 2.497691 | 0.153235 |
| spot_perp_positive_funding_top_1_14d | retail_taker_with_slippage | 10.146755 | 11.250000 | -1.103245 | False | 3.406656 | 0.093076 |
| spot_perp_positive_funding_top_1_7d | low_slippage_taker_like | 5.782354 | 7.500000 | -1.717646 | False | 1.734848 | 0.167991 |
| spot_perp_positive_funding_top_3_14d | expensive_or_thin_execution | 12.475663 | 15.000000 | -2.524337 | False | 4.967699 | 0.077563 |

## Interpretation

A candidate only graduates from historical carry screen to execution watch when it has positive headroom under at least one realistic fee/slippage scenario. The low-turnover 14-day cluster is the only current cluster with meaningful room.
