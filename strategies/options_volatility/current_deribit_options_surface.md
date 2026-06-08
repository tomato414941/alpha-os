# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BTC | 2026-06-09 | 0.82 | 70.62 | 22.64 | 7.71 | 0.1751 | 6413 | 2853221 | put_skew_watch | 47.9724 |
| BTC | 2026-06-12 | 3.82 | 58.03 | 15.02 | 8.87 | 0.0579 | 31691 | 2099327 | put_skew_watch | 43.4671 |
| ETH | 2026-06-09 | 0.82 | 78.50 | 19.57 | 3.77 | 0.1804 | 47296 | 375997 | put_skew_watch | 36.9992 |
| BTC | 2026-06-10 | 1.82 | 62.91 | 19.37 | 2.68 | 0.1066 | 1889 | 842685 | put_skew_watch | 33.7191 |
| ETH | 2026-06-12 | 3.82 | 73.24 | 10.73 | 6.03 | 0.0957 | 146231 | 823007 | put_skew_watch | 33.6791 |
| ETH | 2026-06-26 | 17.82 | 63.71 | 5.56 | 5.93 | 0.0304 | 906726 | 867426 | front_vol_premium_watch | 29.2550 |
| BTC | 2026-06-11 | 2.82 | 60.23 | 15.11 | 2.20 | 0.0725 | 658 | 309082 | put_skew_watch | 27.6742 |
| BTC | 2026-06-19 | 10.82 | 49.16 | 8.65 | 2.98 | 0.0442 | 16416 | 1417257 | put_skew_watch | 24.8884 |
| ETH | 2026-06-19 | 10.82 | 67.21 | 7.32 | 3.50 | 0.0449 | 60392 | 522382 | put_skew_watch | 24.7293 |
| ETH | 2026-06-10 | 1.82 | 74.73 | 14.15 | 0.56 | 0.1235 | 15094 | 269067 | put_skew_watch | 24.6318 |
| BTC | 2026-06-26 | 17.82 | 46.18 | 6.84 | 2.95 | 0.0324 | 144696 | 5364711 | put_skew_watch | 24.5653 |
| ETH | 2026-06-11 | 2.82 | 74.17 | 12.52 | 0.93 | 0.0887 | 7442 | 230658 | put_skew_watch | 23.4373 |
| ETH | 2026-09-25 | 108.82 | 57.66 | 1.76 | -1.78 | 0.0180 | 293708 | 575230 | surface_context | 16.5118 |
| BTC | 2026-09-25 | 108.82 | 42.68 | 2.00 | -1.59 | 0.0173 | 76294 | 2934265 | surface_context | 16.4954 |
| BTC | 2026-07-31 | 52.82 | 43.23 | 3.77 | 0.22 | 0.0161 | 38244 | 2200394 | surface_context | 15.1029 |
| ETH | 2026-12-25 | 199.82 | 59.44 | 1.19 | -1.33 | 0.0226 | 397195 | 456545 | surface_context | 15.0633 |
| BTC | 2026-12-25 | 199.82 | 44.27 | 1.53 | -0.69 | 0.0161 | 80128 | 6181332 | surface_context | 14.5727 |
| ETH | 2026-07-31 | 52.82 | 57.78 | 2.69 | 0.34 | 0.0290 | 143351 | 1100356 | surface_context | 14.5099 |
| BTC | 2026-08-28 | 80.82 | 43.01 | 2.61 | 0.33 | 0.0199 | 6888 | 854303 | surface_context | 13.0001 |
| ETH | 2026-08-28 | 80.82 | 57.44 | 1.81 | -0.22 | 0.0202 | 17636 | 209915 | surface_context | 11.7781 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
