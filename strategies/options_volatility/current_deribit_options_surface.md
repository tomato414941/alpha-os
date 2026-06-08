# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ETH | 2026-06-12 | 3.81 | 72.97 | 11.93 | 8.54 | 0.0547 | 146441 | 827503 | put_skew_watch | 39.9840 |
| BTC | 2026-06-12 | 3.81 | 55.79 | 14.95 | 6.95 | 0.0638 | 31580 | 2099131 | put_skew_watch | 39.5439 |
| BTC | 2026-06-09 | 0.81 | 64.82 | 19.30 | 5.02 | 0.3108 | 6361 | 2846881 | put_skew_watch | 38.9763 |
| ETH | 2026-06-09 | 0.81 | 77.33 | 16.98 | 2.57 | 0.1322 | 46948 | 373373 | put_skew_watch | 32.0995 |
| BTC | 2026-06-10 | 1.81 | 59.80 | 17.89 | 1.78 | 0.0962 | 1845 | 838191 | put_skew_watch | 30.4473 |
| BTC | 2026-06-11 | 2.81 | 58.02 | 15.21 | 2.23 | 0.0835 | 662 | 311623 | put_skew_watch | 27.8181 |
| BTC | 2026-06-26 | 17.81 | 45.99 | 7.17 | 3.55 | 0.0333 | 144680 | 5166484 | put_skew_watch | 26.0771 |
| ETH | 2026-06-10 | 1.81 | 74.76 | 15.10 | 0.64 | 0.0966 | 14231 | 270184 | put_skew_watch | 25.7717 |
| ETH | 2026-06-26 | 17.81 | 61.65 | 5.70 | 3.92 | 0.0252 | 907234 | 874560 | put_skew_watch | 25.3890 |
| BTC | 2026-06-19 | 10.81 | 48.84 | 9.19 | 2.85 | 0.0460 | 16630 | 1849683 | put_skew_watch | 25.2860 |
| ETH | 2026-06-11 | 2.81 | 74.12 | 13.85 | 1.15 | 0.0775 | 7911 | 223652 | put_skew_watch | 25.2429 |
| ETH | 2026-06-19 | 10.81 | 64.43 | 7.43 | 2.78 | 0.0359 | 60081 | 524626 | put_skew_watch | 23.4167 |
| BTC | 2026-09-25 | 108.81 | 42.58 | 2.50 | -1.49 | 0.0196 | 76297 | 2943627 | surface_context | 16.7921 |
| ETH | 2026-09-25 | 108.81 | 57.54 | 1.77 | -1.83 | 0.0137 | 293708 | 570752 | surface_context | 16.6271 |
| BTC | 2026-07-31 | 52.81 | 42.44 | 4.13 | -0.07 | 0.0188 | 38270 | 2324207 | surface_context | 15.1815 |
| ETH | 2026-12-25 | 199.81 | 59.37 | 1.30 | -1.22 | 0.0140 | 397195 | 456545 | surface_context | 14.9705 |
| BTC | 2026-12-25 | 199.81 | 44.07 | 1.11 | -0.89 | 0.0158 | 80458 | 10039779 | surface_context | 14.7657 |
| ETH | 2026-07-31 | 52.81 | 57.73 | 2.66 | 0.33 | 0.0177 | 143311 | 1109120 | surface_context | 14.4858 |
| BTC | 2026-08-28 | 80.81 | 42.51 | 2.38 | -0.07 | 0.0195 | 6879 | 867514 | surface_context | 12.2568 |
| ETH | 2026-08-28 | 80.81 | 57.40 | 2.25 | -0.14 | 0.0146 | 17858 | 222694 | surface_context | 12.1003 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
