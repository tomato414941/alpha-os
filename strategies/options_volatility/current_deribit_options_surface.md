# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BTC | 2026-06-12 | 3.70 | 48.19 | 13.80 | 3.59 | 0.0868 | 31640 | 2636980 | put_skew_watch | 31.7278 |
| BTC | 2026-06-10 | 1.70 | 48.10 | 18.05 | -1.46 | 0.1235 | 4171 | 1253581 | put_skew_watch | 30.4415 |
| BTC | 2026-06-09 | 0.70 | 48.63 | 19.58 | 0.53 | 0.3122 | 6757 | 3056423 | put_skew_watch | 30.3306 |
| ETH | 2026-06-09 | 0.70 | 61.72 | 14.11 | -3.09 | 0.1821 | 49603 | 448948 | put_skew_watch | 30.2736 |
| ETH | 2026-06-12 | 3.70 | 65.68 | 9.43 | 4.24 | 0.0862 | 146954 | 1008743 | put_skew_watch | 28.9085 |
| BTC | 2026-06-11 | 2.70 | 49.56 | 14.50 | 1.37 | 0.1044 | 2716 | 993745 | put_skew_watch | 26.4626 |
| ETH | 2026-06-10 | 1.70 | 64.81 | 13.86 | -1.52 | 0.1250 | 16544 | 320817 | put_skew_watch | 26.3749 |
| ETH | 2026-06-26 | 17.70 | 59.47 | 5.98 | 2.72 | 0.0296 | 909097 | 1337148 | put_skew_watch | 23.4456 |
| BTC | 2026-06-19 | 10.70 | 44.60 | 7.65 | 2.27 | 0.0603 | 16962 | 2822180 | put_skew_watch | 22.7496 |
| ETH | 2026-06-19 | 10.70 | 61.44 | 7.30 | 1.97 | 0.0402 | 60927 | 638260 | put_skew_watch | 21.7493 |
| ETH | 2026-06-11 | 2.70 | 66.33 | 11.05 | 0.65 | 0.0992 | 8294 | 273728 | put_skew_watch | 21.5077 |
| BTC | 2026-06-26 | 17.70 | 42.33 | 6.06 | 1.01 | 0.0474 | 144993 | 6106196 | put_skew_watch | 19.9324 |
| BTC | 2026-09-25 | 108.70 | 41.52 | 1.77 | -2.16 | 0.0178 | 76331 | 3268180 | surface_context | 17.4514 |
| ETH | 2026-09-25 | 108.70 | 57.08 | 1.63 | -1.98 | 0.0142 | 294733 | 814026 | surface_context | 16.9416 |
| ETH | 2026-12-25 | 199.70 | 59.06 | 1.28 | -1.40 | 0.0190 | 397009 | 537794 | surface_context | 15.3715 |
| BTC | 2026-07-31 | 52.70 | 41.32 | 3.46 | -0.22 | 0.0219 | 38652 | 3346974 | surface_context | 14.9681 |
| BTC | 2026-12-25 | 199.70 | 43.68 | 1.27 | -0.88 | 0.0129 | 80489 | 10637977 | surface_context | 14.9368 |
| ETH | 2026-07-31 | 52.70 | 56.75 | 3.31 | -0.11 | 0.0206 | 143532 | 1237059 | surface_context | 14.7382 |
| BTC | 2026-08-28 | 80.70 | 41.54 | 2.79 | 0.02 | 0.0237 | 6990 | 1383826 | surface_context | 12.7682 |
| ETH | 2026-08-28 | 80.70 | 56.86 | 2.30 | -0.22 | 0.0160 | 17719 | 282859 | surface_context | 12.4080 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
