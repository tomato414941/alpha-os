# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ETH | 2026-06-09 | 0.92 | 72.63 | 31.05 | -1.31 | 0.1625 | 43859 | 314766 | put_skew_watch | 43.4851 |
| BTC | 2026-06-12 | 3.92 | 55.10 | 18.82 | 5.67 | 0.0736 | 31558 | 2023687 | put_skew_watch | 40.8181 |
| ETH | 2026-06-12 | 3.92 | 74.19 | 16.26 | 5.95 | 0.0581 | 144927 | 728104 | put_skew_watch | 39.0672 |
| BTC | 2026-06-09 | 0.92 | 58.68 | 27.54 | 1.07 | 0.2552 | 4396 | 910977 | put_skew_watch | 38.7724 |
| ETH | 2026-06-10 | 1.92 | 73.94 | 24.45 | -0.63 | 0.1062 | 14282 | 218258 | put_skew_watch | 34.9915 |
| BTC | 2026-06-10 | 1.92 | 57.61 | 24.24 | 0.81 | 0.1186 | 1759 | 760414 | put_skew_watch | 34.7495 |
| BTC | 2026-06-11 | 2.92 | 56.80 | 20.14 | 1.70 | 0.1081 | 524 | 285717 | put_skew_watch | 31.4998 |
| ETH | 2026-06-11 | 2.92 | 74.57 | 18.58 | 0.38 | 0.0793 | 6256 | 183738 | put_skew_watch | 28.2420 |
| ETH | 2026-06-26 | 17.92 | 64.41 | 5.65 | 5.18 | 0.0269 | 905203 | 695938 | put_skew_watch | 27.7555 |
| BTC | 2026-06-26 | 17.92 | 46.61 | 8.17 | 3.31 | 0.0255 | 144520 | 4879281 | put_skew_watch | 26.5873 |
| BTC | 2026-06-19 | 10.92 | 49.43 | 10.50 | 2.82 | 0.0421 | 16359 | 1240210 | put_skew_watch | 26.3631 |
| ETH | 2026-06-19 | 10.92 | 68.24 | 7.40 | 3.83 | 0.0397 | 58415 | 425925 | put_skew_watch | 25.3766 |
| BTC | 2026-09-25 | 108.92 | 43.16 | 2.50 | -1.31 | 0.0194 | 76271 | 2913396 | surface_context | 16.4280 |
| ETH | 2026-09-25 | 108.92 | 58.01 | 1.69 | -1.65 | 0.0167 | 293641 | 578439 | surface_context | 16.1866 |
| BTC | 2026-07-31 | 52.92 | 43.30 | 4.26 | 0.23 | 0.0194 | 38103 | 1256254 | surface_context | 15.3612 |
| ETH | 2026-07-31 | 52.92 | 59.23 | 2.64 | 0.71 | 0.0193 | 143104 | 1025253 | surface_context | 15.1878 |
| ETH | 2026-12-25 | 199.92 | 59.66 | 1.28 | -1.23 | 0.0201 | 396904 | 328512 | surface_context | 14.8151 |
| BTC | 2026-12-25 | 199.92 | 44.47 | 1.15 | -0.65 | 0.0126 | 80128 | 6093526 | surface_context | 14.1135 |
| ETH | 2026-08-28 | 80.92 | 58.52 | 1.74 | 0.51 | 0.0140 | 17181 | 137935 | surface_context | 12.1068 |
| BTC | 2026-08-28 | 80.92 | 43.07 | 2.36 | -0.09 | 0.0197 | 6814 | 582124 | surface_context | 12.0990 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
