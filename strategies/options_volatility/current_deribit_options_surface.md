# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BTC | 2026-06-08 | 0.49 | 75.87 | 32.77 | -7.43 | 0.2361 | 5056 | 959012 | put_skew_watch | 56.8435 |
| BTC | 2026-06-12 | 4.49 | 68.36 | 18.74 | 12.42 | 0.0619 | 30593 | 1189878 | put_skew_watch | 54.0173 |
| BTC | 2026-06-09 | 1.49 | 83.30 | 23.22 | 5.49 | 0.0801 | 3273 | 981392 | put_skew_watch | 43.5467 |
| BTC | 2026-06-10 | 2.49 | 77.81 | 21.50 | 5.01 | 0.0662 | 1104 | 557453 | put_skew_watch | 40.1770 |
| ETH | 2026-06-08 | 0.49 | 89.46 | 27.41 | -0.43 | 0.1774 | 52708 | 284011 | put_skew_watch | 38.0904 |
| ETH | 2026-06-12 | 4.49 | 81.25 | 9.65 | 8.45 | 0.0546 | 138292 | 340811 | put_skew_watch | 37.1142 |
| BTC | 2026-06-11 | 3.49 | 72.80 | 18.38 | 4.44 | 0.0530 | 139 | 153713 | put_skew_watch | 34.4879 |
| ETH | 2026-06-09 | 1.49 | 89.89 | 14.05 | 4.07 | 0.0761 | 33639 | 137371 | put_skew_watch | 31.7025 |
| BTC | 2026-06-19 | 11.49 | 55.94 | 10.76 | 4.71 | 0.0289 | 15816 | 872866 | put_skew_watch | 30.2624 |
| ETH | 2026-06-26 | 18.49 | 68.30 | 4.55 | 6.74 | 0.0297 | 904955 | 144610 | front_vol_premium_watch | 29.0874 |
| BTC | 2026-06-26 | 18.49 | 51.23 | 6.43 | 5.12 | 0.0261 | 143565 | 2835496 | put_skew_watch | 28.2275 |
| ETH | 2026-06-10 | 2.49 | 85.82 | 11.62 | 2.57 | 0.0480 | 4914 | 80952 | put_skew_watch | 25.2637 |
| ETH | 2026-06-19 | 11.49 | 72.80 | 6.42 | 4.50 | 0.0351 | 54429 | 99504 | put_skew_watch | 25.0834 |
| ETH | 2026-06-11 | 3.49 | 83.25 | 9.61 | 2.00 | 0.0567 | 1029 | 39702 | put_skew_watch | 21.1083 |
| BTC | 2026-07-31 | 53.49 | 46.11 | 3.37 | 1.16 | 0.0182 | 37201 | 1370647 | surface_context | 16.3611 |
| ETH | 2026-07-31 | 53.49 | 61.56 | 2.17 | 1.83 | 0.0224 | 139528 | 123060 | surface_context | 16.0199 |
| ETH | 2026-09-25 | 109.49 | 59.93 | 1.96 | -0.86 | 0.0205 | 293162 | 187113 | surface_context | 14.3783 |
| BTC | 2026-09-25 | 109.49 | 44.76 | 1.63 | -0.89 | 0.0186 | 76382 | 1066107 | surface_context | 14.2836 |
| BTC | 2026-12-25 | 200.49 | 45.65 | 1.31 | -0.24 | 0.0206 | 80434 | 1533511 | surface_context | 12.8400 |
| BTC | 2026-08-28 | 81.49 | 44.95 | 2.52 | 0.19 | 0.0238 | 6624 | 339403 | surface_context | 12.2042 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
