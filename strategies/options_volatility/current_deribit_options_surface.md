# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ETH | 2026-06-09 | 0.70 | 61.17 | 19.46 | -3.49 | 0.2355 | 49206 | 433040 | put_skew_watch | 36.2976 |
| BTC | 2026-06-12 | 3.70 | 48.46 | 13.19 | 3.71 | 0.0804 | 31668 | 2635174 | put_skew_watch | 31.3706 |
| ETH | 2026-06-12 | 3.70 | 65.60 | 11.81 | 4.11 | 0.0990 | 147173 | 1008202 | put_skew_watch | 31.0035 |
| BTC | 2026-06-09 | 0.70 | 49.04 | 19.75 | 0.27 | 0.1231 | 6676 | 3040139 | put_skew_watch | 30.3512 |
| BTC | 2026-06-10 | 1.70 | 48.77 | 17.95 | -1.23 | 0.1470 | 4163 | 1268821 | put_skew_watch | 29.8389 |
| ETH | 2026-06-10 | 1.70 | 64.66 | 17.01 | -1.61 | 0.0963 | 16341 | 316669 | put_skew_watch | 29.7513 |
| BTC | 2026-06-11 | 2.70 | 50.00 | 14.70 | 1.54 | 0.1089 | 2716 | 993680 | put_skew_watch | 26.9936 |
| ETH | 2026-06-11 | 2.70 | 66.27 | 13.85 | 0.67 | 0.0831 | 8232 | 272075 | put_skew_watch | 24.3740 |
| BTC | 2026-06-19 | 10.70 | 44.75 | 7.57 | 2.32 | 0.0442 | 16962 | 2821556 | put_skew_watch | 22.8016 |
| ETH | 2026-06-26 | 17.70 | 59.29 | 5.55 | 2.53 | 0.0319 | 908660 | 1305058 | put_skew_watch | 22.6203 |
| ETH | 2026-06-19 | 10.70 | 61.49 | 7.07 | 2.20 | 0.0476 | 60896 | 626054 | put_skew_watch | 21.9561 |
| BTC | 2026-06-26 | 17.70 | 42.43 | 6.04 | 1.11 | 0.0343 | 144979 | 6085547 | put_skew_watch | 20.1370 |
| BTC | 2026-09-25 | 108.70 | 41.64 | 1.74 | -2.08 | 0.0160 | 76304 | 3169023 | surface_context | 17.2515 |
| ETH | 2026-09-25 | 108.70 | 57.10 | 1.60 | -2.00 | 0.0161 | 294388 | 640412 | surface_context | 16.8431 |
| ETH | 2026-12-25 | 199.70 | 59.10 | 1.17 | -1.36 | 0.0141 | 396909 | 523514 | surface_context | 15.1794 |
| BTC | 2026-07-31 | 52.70 | 41.32 | 3.46 | -0.28 | 0.0268 | 38640 | 3324898 | surface_context | 15.0752 |
| BTC | 2026-12-25 | 199.70 | 43.72 | 1.23 | -0.88 | 0.0117 | 80484 | 10635929 | surface_context | 14.8991 |
| ETH | 2026-07-31 | 52.70 | 56.76 | 3.20 | -0.11 | 0.0192 | 143403 | 1232025 | surface_context | 14.6289 |
| BTC | 2026-08-28 | 80.70 | 41.60 | 2.74 | -0.04 | 0.0173 | 6990 | 1385421 | surface_context | 12.7715 |
| ETH | 2026-08-28 | 80.70 | 56.87 | 2.20 | -0.23 | 0.0149 | 17719 | 283016 | surface_context | 12.3306 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
