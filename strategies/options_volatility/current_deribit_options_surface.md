# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BTC | 2026-06-08 | 0.28 | 68.60 | 36.21 | -6.06 | 0.3111 | 5326 | 1128156 | put_skew_watch | 57.4866 |
| ETH | 2026-06-08 | 0.28 | 87.62 | 36.82 | -1.50 | 0.3809 | 54206 | 384356 | put_skew_watch | 49.3770 |
| BTC | 2026-06-12 | 4.28 | 62.26 | 14.19 | 10.28 | 0.0606 | 31193 | 2026541 | put_skew_watch | 45.4297 |
| ETH | 2026-06-12 | 4.28 | 79.01 | 11.01 | 10.63 | 0.0555 | 141322 | 548808 | put_skew_watch | 43.0485 |
| BTC | 2026-06-09 | 1.28 | 74.66 | 23.92 | 3.73 | 0.1077 | 3485 | 990017 | put_skew_watch | 40.7025 |
| ETH | 2026-06-09 | 1.28 | 89.12 | 23.12 | 3.75 | 0.0986 | 36504 | 247400 | put_skew_watch | 40.3786 |
| BTC | 2026-06-10 | 2.28 | 70.93 | 18.98 | 4.89 | 0.0859 | 1366 | 776393 | put_skew_watch | 37.6143 |
| ETH | 2026-06-10 | 2.28 | 85.37 | 16.22 | 3.47 | 0.0940 | 10299 | 206113 | put_skew_watch | 32.2989 |
| BTC | 2026-06-11 | 3.28 | 66.04 | 14.66 | 3.78 | 0.0809 | 298 | 277138 | put_skew_watch | 29.9761 |
| ETH | 2026-06-11 | 3.28 | 81.90 | 12.94 | 2.89 | 0.0723 | 4338 | 115757 | put_skew_watch | 27.2764 |
| ETH | 2026-06-26 | 18.28 | 64.92 | 4.77 | 5.42 | 0.0308 | 904420 | 543296 | front_vol_premium_watch | 27.2397 |
| BTC | 2026-06-19 | 11.28 | 51.98 | 8.18 | 3.76 | 0.0417 | 16243 | 1231162 | put_skew_watch | 25.9177 |
| BTC | 2026-06-26 | 18.28 | 48.22 | 6.18 | 3.80 | 0.0327 | 143756 | 3492665 | put_skew_watch | 25.4153 |
| ETH | 2026-06-19 | 11.28 | 68.38 | 5.89 | 3.46 | 0.0296 | 55602 | 181755 | put_skew_watch | 22.7554 |
| ETH | 2026-09-25 | 109.28 | 58.61 | 1.60 | -1.31 | 0.0133 | 292907 | 368904 | surface_context | 15.2270 |
| BTC | 2026-07-31 | 53.28 | 44.42 | 3.41 | 0.42 | 0.0182 | 37259 | 1591606 | surface_context | 14.9867 |
| BTC | 2026-09-25 | 109.28 | 43.78 | 1.85 | -1.10 | 0.0193 | 76390 | 1196667 | surface_context | 14.9725 |
| ETH | 2026-07-31 | 53.28 | 59.50 | 2.30 | 0.74 | 0.0154 | 140262 | 286603 | surface_context | 14.3535 |
| ETH | 2026-12-25 | 200.28 | 59.92 | 1.11 | -1.16 | 0.0178 | 394800 | 109889 | surface_context | 14.0318 |
| BTC | 2026-12-25 | 200.28 | 44.88 | 1.11 | -0.52 | 0.0229 | 80436 | 2091087 | surface_context | 13.3300 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
