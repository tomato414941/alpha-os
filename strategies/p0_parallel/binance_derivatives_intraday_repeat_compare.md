# Binance Derivatives Intraday Repeat Compare

This compares a prior non-overlapping 5m-to-1h label window against the recent window. Rows with the same preferred bucket across windows are repeat candidates, not trade instructions.

| symbol | feature | status | prior bucket | recent bucket | prior score | recent score | combined score | next step |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| ARBUSDT | count_long_short_ratio | intraday_repeat_watch | low | low | 135.9200 | 363.7439 | 365.8527 | run ARBUSDT count_long_short_ratio intraday paper label with fees, spread, funding PnL, fill assumptions, and stop behavior |
| ARBUSDT | count_top_long_short_ratio | intraday_repeat_watch | low | low | 115.7972 | 253.4757 | 319.6518 | run ARBUSDT count_top_long_short_ratio intraday paper label with fees, spread, funding PnL, fill assumptions, and stop behavior |
| ARBUSDT | sum_top_long_short_ratio | intraday_repeat_watch | low | low | 120.7376 | 187.4846 | 312.3820 | run ARBUSDT sum_top_long_short_ratio intraday paper label with fees, spread, funding PnL, fill assumptions, and stop behavior |
| ADAUSDT | count_long_short_ratio | intraday_repeat_watch | low | low | 96.0980 | 201.6159 | 285.6408 | run ADAUSDT count_long_short_ratio intraday paper label with fees, spread, funding PnL, fill assumptions, and stop behavior |
| UNIUSDT | abs_premium_close | intraday_repeat_watch | high | high | 107.5477 | 101.4635 | 273.2658 | run UNIUSDT abs_premium_close intraday paper label with fees, spread, funding PnL, fill assumptions, and stop behavior |
| UNIUSDT | premium_close | intraday_repeat_watch | low | low | 107.1757 | 95.2261 | 265.7064 | run UNIUSDT premium_close intraday paper label with fees, spread, funding PnL, fill assumptions, and stop behavior |
| OPUSDT | count_top_long_short_ratio | intraday_bucket_shift | high | low | 239.1800 | 287.4978 | 264.5156 | explain OPUSDT count_top_long_short_ratio bucket shift by regime before any promotion |
| DOGEUSDT | premium_close | intraday_repeat_watch | low | low | 87.4663 | 132.8595 | 261.5314 | run DOGEUSDT premium_close intraday paper label with fees, spread, funding PnL, fill assumptions, and stop behavior |
| OPUSDT | sum_top_long_short_ratio | intraday_bucket_shift | low | high | 310.3726 | 231.1777 | 259.4878 | explain OPUSDT sum_top_long_short_ratio bucket shift by regime before any promotion |
| DOGEUSDT | abs_premium_close | intraday_repeat_watch | high | high | 85.6391 | 128.5846 | 258.4838 | run DOGEUSDT abs_premium_close intraday paper label with fees, spread, funding PnL, fill assumptions, and stop behavior |
| OPUSDT | count_long_short_ratio | intraday_bucket_shift | high | low | 213.8113 | 367.0815 | 249.9898 | explain OPUSDT count_long_short_ratio bucket shift by regime before any promotion |
| NEARUSDT | count_long_short_ratio | intraday_bucket_shift | low_mean_only | high | 195.8947 | 298.4522 | 234.7641 | explain NEARUSDT count_long_short_ratio bucket shift by regime before any promotion |
| NEARUSDT | count_top_long_short_ratio | intraday_bucket_shift | low_mean_only | high | 163.8587 | 275.1492 | 191.6603 | explain NEARUSDT count_top_long_short_ratio bucket shift by regime before any promotion |
| NEARUSDT | oi_value_change | intraday_recent_watch | low | low | 79.8791 | 176.1331 | 191.0815 | keep NEARUSDT oi_value_change as context until another non-overlapping window repeats |
| NEARUSDT | premium_close | intraday_recent_watch | low | low | 59.7603 | 221.3097 | 175.9743 | keep NEARUSDT premium_close as context until another non-overlapping window repeats |
| DOGEUSDT | sum_top_long_short_ratio | intraday_recent_watch | high | low | 119.9565 | 128.6228 | 169.6724 | keep DOGEUSDT sum_top_long_short_ratio as context until another non-overlapping window repeats |
| NEARUSDT | abs_premium_close | intraday_recent_watch | high | high | 52.5819 | 224.8038 | 168.0590 | keep NEARUSDT abs_premium_close as context until another non-overlapping window repeats |
| ADAUSDT | count_top_long_short_ratio | intraday_recent_watch | low | low | 63.4410 | 157.4848 | 167.6262 | keep ADAUSDT count_top_long_short_ratio as context until another non-overlapping window repeats |
| SOLUSDT | abs_premium_close | intraday_recent_watch | high | high | 61.3406 | 106.2745 | 154.8636 | keep SOLUSDT abs_premium_close as context until another non-overlapping window repeats |
| SOLUSDT | premium_close | intraday_recent_watch | low | low | 60.1368 | 109.2504 | 154.0142 | keep SOLUSDT premium_close as context until another non-overlapping window repeats |
| SOLUSDT | count_long_short_ratio | intraday_recent_watch | low | low | 44.7859 | 160.5407 | 145.8512 | keep SOLUSDT count_long_short_ratio as context until another non-overlapping window repeats |
| NEARUSDT | sum_taker_long_short_vol_ratio | intraday_recent_watch | low | high_mean_only | 102.0048 | 84.8874 | 142.2658 | keep NEARUSDT sum_taker_long_short_vol_ratio as context until another non-overlapping window repeats |
| DOGEUSDT | count_top_long_short_ratio | weak_intraday_repeat_context | low | low | 99.4285 | 66.4190 | 139.5885 | keep DOGEUSDT count_top_long_short_ratio as context until another non-overlapping window repeats |
| BCHUSDT | abs_premium_close | intraday_recent_watch | high | high | 47.0814 | 95.7423 | 135.6461 | keep BCHUSDT abs_premium_close as context until another non-overlapping window repeats |
| SOLUSDT | count_top_long_short_ratio | intraday_recent_watch | low | low | 42.8215 | 116.2760 | 134.6410 | keep SOLUSDT count_top_long_short_ratio as context until another non-overlapping window repeats |
| UNIUSDT | sum_top_long_short_ratio | weak_intraday_repeat_context | high | high | 194.2475 | 28.0871 | 132.5540 | keep UNIUSDT sum_top_long_short_ratio as context until another non-overlapping window repeats |
| BCHUSDT | premium_close | intraday_recent_watch | low | low | 44.2344 | 96.6352 | 132.4083 | keep BCHUSDT premium_close as context until another non-overlapping window repeats |
| OPUSDT | oi_value_change | weak_intraday_repeat_context | high | high | 48.6922 | 40.2486 | 118.0367 | keep OPUSDT oi_value_change as context until another non-overlapping window repeats |
| BCHUSDT | sum_top_long_short_ratio | weak_intraday_repeat_context | high | high_mean_only | 216.6146 | 25.1166 | 113.4628 | keep BCHUSDT sum_top_long_short_ratio as context until another non-overlapping window repeats |
| SOLUSDT | sum_top_long_short_ratio | weak_intraday_repeat_context | high | low | 129.1449 | 72.1187 | 112.3714 | keep SOLUSDT sum_top_long_short_ratio as context until another non-overlapping window repeats |
| ADAUSDT | premium_close | intraday_recent_watch | high | low | 72.9287 | 118.7679 | 111.2680 | keep ADAUSDT premium_close as context until another non-overlapping window repeats |
| DOGEUSDT | oi_value_change | weak_intraday_repeat_context | low | low | 34.2752 | 49.9535 | 111.1209 | keep DOGEUSDT oi_value_change as context until another non-overlapping window repeats |
| ADAUSDT | abs_premium_close | intraday_recent_watch | low | high | 72.2514 | 120.5383 | 110.8094 | keep ADAUSDT abs_premium_close as context until another non-overlapping window repeats |
| NEARUSDT | sum_top_long_short_ratio | intraday_bucket_shift | low_mean_only | high | 74.6587 | 399.6268 | 109.5157 | explain NEARUSDT sum_top_long_short_ratio bucket shift by regime before any promotion |
| ADAUSDT | oi_value_change | weak_intraday_repeat_context | high_mean_only | low | 61.8664 | 68.6077 | 107.9612 | keep ADAUSDT oi_value_change as context until another non-overlapping window repeats |
| ADAUSDT | sum_taker_long_short_vol_ratio | weak_intraday_repeat_context | low | low | 28.2386 | 65.5169 | 106.9897 | keep ADAUSDT sum_taker_long_short_vol_ratio as context until another non-overlapping window repeats |
| UNIUSDT | sum_taker_long_short_vol_ratio | intraday_recent_watch | high | low | 64.1729 | 120.7853 | 101.1646 | keep UNIUSDT sum_taker_long_short_vol_ratio as context until another non-overlapping window repeats |
| OPUSDT | abs_premium_close | weak_intraday_repeat_context | high | high | 22.8499 | 68.5422 | 101.1283 | keep OPUSDT abs_premium_close as context until another non-overlapping window repeats |
| BCHUSDT | count_top_long_short_ratio | weak_intraday_repeat_context | low_mean_only | low_mean_only | 50.0333 | 25.2549 | 100.3125 | keep BCHUSDT count_top_long_short_ratio as context until another non-overlapping window repeats |
| ADAUSDT | sum_top_long_short_ratio | weak_intraday_repeat_context | low | high_mean_only | 82.5751 | 51.2698 | 98.0388 | keep ADAUSDT sum_top_long_short_ratio as context until another non-overlapping window repeats |
| BCHUSDT | oi_value_change | weak_intraday_repeat_context | high_mean_only | high_mean_only | 29.8009 | 26.0698 | 97.2440 | keep BCHUSDT oi_value_change as context until another non-overlapping window repeats |
| BCHUSDT | count_long_short_ratio | weak_intraday_repeat_context | low | low_mean_only | 101.4512 | 28.4139 | 94.3870 | keep BCHUSDT count_long_short_ratio as context until another non-overlapping window repeats |
| DOGEUSDT | sum_taker_long_short_vol_ratio | weak_intraday_repeat_context | low | low | 16.7071 | 25.0827 | 85.0651 | keep DOGEUSDT sum_taker_long_short_vol_ratio as context until another non-overlapping window repeats |
| OPUSDT | premium_close | weak_intraday_repeat_context | high | low | 52.0853 | 65.4265 | 75.5877 | keep OPUSDT premium_close as context until another non-overlapping window repeats |
| ARBUSDT | oi_value_change | weak_intraday_repeat_context | low_mean_only | low | 20.9240 | 52.2957 | 75.5679 | keep ARBUSDT oi_value_change as context until another non-overlapping window repeats |
| SOLUSDT | oi_value_change | weak_intraday_repeat_context | high_mean_only | low | 35.8968 | 47.1863 | 72.5135 | keep SOLUSDT oi_value_change as context until another non-overlapping window repeats |
| SOLUSDT | sum_taker_long_short_vol_ratio | weak_intraday_repeat_context | low_mean_only | low | 16.0995 | 62.8443 | 71.8882 | keep SOLUSDT sum_taker_long_short_vol_ratio as context until another non-overlapping window repeats |
| BCHUSDT | sum_taker_long_short_vol_ratio | weak_intraday_repeat_context | high | low_mean_only | 32.7761 | 49.4010 | 69.2116 | keep BCHUSDT sum_taker_long_short_vol_ratio as context until another non-overlapping window repeats |
| UNIUSDT | oi_value_change | weak_intraday_repeat_context | high_mean_only | low | 27.0432 | 61.1337 | 64.6786 | keep UNIUSDT oi_value_change as context until another non-overlapping window repeats |
| OPUSDT | sum_taker_long_short_vol_ratio | intraday_recent_watch | high | low | 35.9739 | 99.0164 | 62.9720 | keep OPUSDT sum_taker_long_short_vol_ratio as context until another non-overlapping window repeats |
| ARBUSDT | sum_taker_long_short_vol_ratio | weak_intraday_repeat_context | high_mean_only | low | 20.1703 | 77.4744 | 59.6993 | keep ARBUSDT sum_taker_long_short_vol_ratio as context until another non-overlapping window repeats |
| ARBUSDT | premium_close | weak_intraday_repeat_context | high | low | 31.2297 | 77.4591 | 52.9674 | keep ARBUSDT premium_close as context until another non-overlapping window repeats |
| UNIUSDT | count_top_long_short_ratio | weak_intraday_repeat_context | low | high | 45.9788 | 27.9989 | 42.7945 | keep UNIUSDT count_top_long_short_ratio as context until another non-overlapping window repeats |
| DOGEUSDT | count_long_short_ratio | weak_intraday_repeat_context | low_mean_only | high | 80.1488 | 21.2318 | 41.5079 | keep DOGEUSDT count_long_short_ratio as context until another non-overlapping window repeats |
| ARBUSDT | abs_premium_close | weak_intraday_repeat_context | low | high | 20.7533 | 78.9949 | 40.7030 | keep ARBUSDT abs_premium_close as context until another non-overlapping window repeats |
| UNIUSDT | count_long_short_ratio | weak_intraday_repeat_context | low | high | 40.1945 | 16.9816 | 28.4168 | keep UNIUSDT count_long_short_ratio as context until another non-overlapping window repeats |

## Interpretation

`intraday_repeat_priority` rows repeated the same symbol-feature bucket across non-overlapping windows. `intraday_recent_only_priority` rows are current but not repeated. `intraday_bucket_shift` rows changed direction and should not be promoted without a regime explanation.
