# Binance Derivatives Feature Regime Compare

This compares the 2024Q1 Binance USD-M symbol-feature queue with the recent-window queue. It separates candidates that persist across regimes from candidates that appear only in the recent panel.

| symbol | feature | status | historical bucket | recent bucket | historical score | recent score | combined score | next step |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| ARBUSDT | mean_sum_top_long_short_ratio | persistent_symbol_feature | low | low | 268.2077 | 428.4488 | 472.3644 | rerun ARBUSDT mean_sum_top_long_short_ratio with recent intraday labels, then add execution and funding PnL |
| ARBUSDT | oi_value_change | recent_symbol_feature_priority | high_mean_only | high | 15.2622 | 546.0359 | 425.2651 | extend recent ARBUSDT oi_value_change window and check whether the effect survives costs |
| NEARUSDT | mean_funding_rate | persistent_symbol_feature | high_mean_only | high_mean_only | 340.6412 | 305.4067 | 417.7388 | rerun NEARUSDT mean_funding_rate with recent intraday labels, then add execution and funding PnL |
| NEARUSDT | sum_funding_rate | persistent_symbol_feature | high_mean_only | high_mean_only | 340.6412 | 305.4067 | 417.7388 | rerun NEARUSDT sum_funding_rate with recent intraday labels, then add execution and funding PnL |
| BCHUSDT | mean_sum_top_long_short_ratio | persistent_symbol_feature | high | high | 283.1930 | 305.4253 | 397.6440 | rerun BCHUSDT mean_sum_top_long_short_ratio with recent intraday labels, then add execution and funding PnL |
| NEARUSDT | mean_sum_top_long_short_ratio | persistent_symbol_feature | high_mean_only | high_mean_only | 297.5136 | 294.1766 | 395.3446 | rerun NEARUSDT mean_sum_top_long_short_ratio with recent intraday labels, then add execution and funding PnL |
| ARBUSDT | mean_premium_close | recent_symbol_feature_priority | low | high | 28.5578 | 507.4784 | 384.8562 | extend recent ARBUSDT mean_premium_close window and check whether the effect survives costs |
| OPUSDT | oi_value_change | recent_symbol_feature_priority | high | high | 95.7761 | 426.6602 | 375.8508 | extend recent OPUSDT oi_value_change window and check whether the effect survives costs |
| NEARUSDT | mean_premium_close | persistent_symbol_feature | high_mean_only | high | 208.9008 | 288.2721 | 360.4922 | rerun NEARUSDT mean_premium_close with recent intraday labels, then add execution and funding PnL |
| ARBUSDT | mean_funding_rate | recent_symbol_feature_priority | high | high | 13.7478 | 446.2209 | 359.8553 | extend recent ARBUSDT mean_funding_rate window and check whether the effect survives costs |
| ARBUSDT | sum_funding_rate | recent_symbol_feature_priority | high | high | 13.7478 | 446.2209 | 359.8553 | extend recent ARBUSDT sum_funding_rate window and check whether the effect survives costs |
| BCHUSDT | mean_count_top_long_short_ratio | persistent_symbol_feature | low | low | 264.9374 | 253.3369 | 357.3971 | rerun BCHUSDT mean_count_top_long_short_ratio with recent intraday labels, then add execution and funding PnL |
| UNIUSDT | oi_value_change | persistent_symbol_feature | high | high | 328.9493 | 188.8030 | 337.8542 | rerun UNIUSDT oi_value_change with recent intraday labels, then add execution and funding PnL |
| ADAUSDT | mean_count_top_long_short_ratio | recent_symbol_feature_priority | low_mean_only | low | 90.0091 | 359.7898 | 330.3666 | extend recent ADAUSDT mean_count_top_long_short_ratio window and check whether the effect survives costs |
| DOGEUSDT | mean_premium_close | persistent_symbol_feature | high | high | 311.2457 | 183.6070 | 328.2805 | rerun DOGEUSDT mean_premium_close with recent intraday labels, then add execution and funding PnL |
| ADAUSDT | mean_count_long_short_ratio | recent_symbol_feature_priority | low | low | 114.6381 | 342.3502 | 327.6510 | extend recent ADAUSDT mean_count_long_short_ratio window and check whether the effect survives costs |
| ETHUSDT | mean_sum_top_long_short_ratio | persistent_symbol_feature | low | low | 172.2693 | 233.4771 | 312.0543 | rerun ETHUSDT mean_sum_top_long_short_ratio with recent intraday labels, then add execution and funding PnL |
| BCHUSDT | mean_count_long_short_ratio | persistent_symbol_feature | low | low_mean_only | 279.2750 | 172.5488 | 309.9030 | rerun BCHUSDT mean_count_long_short_ratio with recent intraday labels, then add execution and funding PnL |
| BCHUSDT | mean_premium_close | persistent_symbol_feature | high | high | 289.2663 | 163.2299 | 307.3426 | rerun BCHUSDT mean_premium_close with recent intraday labels, then add execution and funding PnL |
| FILUSDT | mean_funding_rate | recent_symbol_feature_priority | high_mean_only | high | 45.2695 | 346.3580 | 305.9771 | extend recent FILUSDT mean_funding_rate window and check whether the effect survives costs |
| FILUSDT | sum_funding_rate | recent_symbol_feature_priority | high_mean_only | high | 45.2695 | 346.3580 | 305.9771 | extend recent FILUSDT sum_funding_rate window and check whether the effect survives costs |
| ARBUSDT | max_abs_premium_close | recent_symbol_feature_priority | high_mean_only | low | 11.5940 | 394.0660 | 305.2008 | extend recent ARBUSDT max_abs_premium_close window and check whether the effect survives costs |
| SOLUSDT | mean_sum_taker_long_short_vol_ratio | persistent_symbol_feature | high | high | 190.2968 | 211.2734 | 303.9316 | rerun SOLUSDT mean_sum_taker_long_short_vol_ratio with recent intraday labels, then add execution and funding PnL |
| ADAUSDT | max_abs_premium_close | bucket_regime_shift | high | low | 143.6566 | 331.9168 | 291.0257 | split ADAUSDT max_abs_premium_close by market regime before using the feature direction |
| BNBUSDT | sum_funding_rate | persistent_symbol_feature | high | high | 197.9477 | 180.5419 | 286.6339 | rerun BNBUSDT sum_funding_rate with recent intraday labels, then add execution and funding PnL |
| BNBUSDT | mean_funding_rate | persistent_symbol_feature | high | high | 197.9477 | 180.5419 | 286.6339 | rerun BNBUSDT mean_funding_rate with recent intraday labels, then add execution and funding PnL |
| AVAXUSDT | mean_sum_taker_long_short_vol_ratio | recent_symbol_feature_priority | low | high | 97.4545 | 317.5573 | 285.5213 | extend recent AVAXUSDT mean_sum_taker_long_short_vol_ratio window and check whether the effect survives costs |
| FILUSDT | mean_count_top_long_short_ratio | persistent_symbol_feature | low | low | 176.9074 | 180.9929 | 279.5630 | rerun FILUSDT mean_count_top_long_short_ratio with recent intraday labels, then add execution and funding PnL |
| DOTUSDT | oi_value_change | recent_symbol_feature_priority | high | high | 60.0748 | 292.3939 | 276.0822 | extend recent DOTUSDT oi_value_change window and check whether the effect survives costs |
| INJUSDT | mean_premium_close | persistent_symbol_feature | low | low | 136.9941 | 181.3146 | 265.8024 | rerun INJUSDT mean_premium_close with recent intraday labels, then add execution and funding PnL |
| ETHUSDT | mean_count_top_long_short_ratio | persistent_symbol_feature | low | low | 199.5467 | 142.7414 | 262.6233 | rerun ETHUSDT mean_count_top_long_short_ratio with recent intraday labels, then add execution and funding PnL |
| DOGEUSDT | mean_count_long_short_ratio | persistent_symbol_feature | low | low | 171.9619 | 154.9582 | 260.9095 | rerun DOGEUSDT mean_count_long_short_ratio with recent intraday labels, then add execution and funding PnL |
| ETHUSDT | mean_count_long_short_ratio | persistent_symbol_feature | low | low | 191.4690 | 137.8862 | 256.6402 | rerun ETHUSDT mean_count_long_short_ratio with recent intraday labels, then add execution and funding PnL |
| LINKUSDT | mean_count_top_long_short_ratio | persistent_symbol_feature | low | low | 165.3431 | 149.5496 | 255.0774 | rerun LINKUSDT mean_count_top_long_short_ratio with recent intraday labels, then add execution and funding PnL |
| FILUSDT | max_abs_premium_close | bucket_regime_shift | high | low | 173.7612 | 251.1102 | 249.0380 | split FILUSDT max_abs_premium_close by market regime before using the feature direction |
| LINKUSDT | mean_premium_close | persistent_symbol_feature | low | low | 126.3242 | 160.1022 | 248.2799 | rerun LINKUSDT mean_premium_close with recent intraday labels, then add execution and funding PnL |
| LINKUSDT | mean_sum_taker_long_short_vol_ratio | persistent_symbol_feature | high | high | 129.6092 | 152.6208 | 244.5668 | rerun LINKUSDT mean_sum_taker_long_short_vol_ratio with recent intraday labels, then add execution and funding PnL |
| FILUSDT | mean_sum_taker_long_short_vol_ratio | recent_symbol_feature_priority | high | high | 13.0081 | 263.1201 | 240.5809 | extend recent FILUSDT mean_sum_taker_long_short_vol_ratio window and check whether the effect survives costs |
| DOGEUSDT | oi_value_change | recent_symbol_feature_priority | high | high | 109.2233 | 211.3032 | 240.5752 | extend recent DOGEUSDT oi_value_change window and check whether the effect survives costs |
| OPUSDT | mean_premium_close | bucket_regime_shift | low | high | 182.3920 | 224.1780 | 234.5529 | split OPUSDT mean_premium_close by market regime before using the feature direction |
| ETCUSDT | oi_value_change | recent_symbol_feature_priority | low | high | 7.1125 | 284.8460 | 232.6393 | extend recent ETCUSDT oi_value_change window and check whether the effect survives costs |
| BNBUSDT | max_abs_premium_close | recent_symbol_feature_priority | low | low | 23.2906 | 239.1902 | 228.6253 | extend recent BNBUSDT max_abs_premium_close window and check whether the effect survives costs |
| NEARUSDT | oi_value_change | bucket_regime_shift | low_mean_only | high_mean_only | 169.2721 | 207.3884 | 219.0477 | split NEARUSDT oi_value_change by market regime before using the feature direction |
| APTUSDT | oi_value_change | bucket_regime_shift | low | high | 224.7165 | 158.8180 | 206.8825 | split APTUSDT oi_value_change by market regime before using the feature direction |
| SOLUSDT | max_abs_premium_close | bucket_regime_shift | high | low | 189.5597 | 172.6243 | 203.5517 | split SOLUSDT max_abs_premium_close by market regime before using the feature direction |
| BCHUSDT | oi_value_change | recent_symbol_feature_priority | high_mean_only | high | 43.3813 | 188.4911 | 202.7027 | extend recent BCHUSDT oi_value_change window and check whether the effect survives costs |
| ADAUSDT | sum_funding_rate | recent_symbol_feature_priority | high | high | 46.5099 | 184.5556 | 201.2396 | extend recent ADAUSDT sum_funding_rate window and check whether the effect survives costs |
| ADAUSDT | mean_funding_rate | recent_symbol_feature_priority | high | high | 46.5099 | 184.5556 | 201.2396 | extend recent ADAUSDT mean_funding_rate window and check whether the effect survives costs |
| DOGEUSDT | max_abs_premium_close | bucket_regime_shift | high | low | 209.9866 | 153.7104 | 198.4071 | split DOGEUSDT max_abs_premium_close by market regime before using the feature direction |
| DOTUSDT | mean_sum_taker_long_short_vol_ratio | bucket_regime_shift | low_mean_only | high | 132.5885 | 193.1976 | 196.9844 | split DOTUSDT mean_sum_taker_long_short_vol_ratio by market regime before using the feature direction |
| LTCUSDT | mean_count_top_long_short_ratio | recent_symbol_feature_priority | low | low | 32.6455 | 182.4223 | 195.0004 | extend recent LTCUSDT mean_count_top_long_short_ratio window and check whether the effect survives costs |
| INJUSDT | mean_sum_taker_long_short_vol_ratio | recent_symbol_feature_priority | low | high | 61.7126 | 196.3648 | 194.2365 | extend recent INJUSDT mean_sum_taker_long_short_vol_ratio window and check whether the effect survives costs |
| FILUSDT | mean_premium_close | recent_symbol_feature_priority | low | high | 13.4544 | 219.9543 | 192.6793 | extend recent FILUSDT mean_premium_close window and check whether the effect survives costs |
| APTUSDT | mean_premium_close | recent_symbol_feature_watch | high | high | 95.5675 | 177.1685 | 188.6082 | extend recent APTUSDT mean_premium_close window and check whether the effect survives costs |
| BCHUSDT | max_abs_premium_close | recent_symbol_feature_priority | high_mean_only | low | 28.6217 | 202.9190 | 186.9150 | extend recent BCHUSDT max_abs_premium_close window and check whether the effect survives costs |
| AVAXUSDT | oi_value_change | bucket_regime_shift | low | high | 123.1022 | 169.4081 | 178.2011 | split AVAXUSDT oi_value_change by market regime before using the feature direction |
| DOGEUSDT | mean_funding_rate | historical_symbol_feature_only | high | high | 300.5611 | 109.8197 | 176.5792 | deprioritize DOGEUSDT mean_funding_rate until it reappears in recent data |
| DOGEUSDT | sum_funding_rate | historical_symbol_feature_only | high | high | 300.5611 | 109.8197 | 176.5792 | deprioritize DOGEUSDT sum_funding_rate until it reappears in recent data |
| ETHUSDT | mean_premium_close | recent_symbol_feature_watch | low | low | 110.6551 | 144.7100 | 172.7908 | extend recent ETHUSDT mean_premium_close window and check whether the effect survives costs |
| INJUSDT | mean_sum_top_long_short_ratio | bucket_regime_shift | low | high | 144.1053 | 147.9866 | 171.6282 | split INJUSDT mean_sum_top_long_short_ratio by market regime before using the feature direction |

## Interpretation

`persistent_symbol_feature` rows deserve the next recent-data rerun and execution check. `recent_symbol_feature_priority` rows are newer regime candidates. `bucket_regime_shift` rows may still matter, but the direction changed and should not be promoted blindly.
