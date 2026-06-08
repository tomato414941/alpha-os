# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BTC | 2026-06-12 | 4.30 | 61.14 | 12.68 | 9.75 | 0.0716 | 31119 | 2061821 | put_skew_watch | 42.8441 |
| BTC | 2026-06-08 | 0.30 | 64.34 | 14.96 | -9.12 | 0.3783 | 5291 | 1129768 | put_skew_watch | 42.2200 |
| ETH | 2026-06-08 | 0.30 | 81.27 | 18.75 | -5.68 | 0.2859 | 54048 | 379550 | put_skew_watch | 39.8503 |
| ETH | 2026-06-12 | 4.30 | 77.83 | 9.87 | 9.34 | 0.0539 | 141324 | 550491 | put_skew_watch | 39.3332 |
| BTC | 2026-06-09 | 1.30 | 73.46 | 23.81 | 3.03 | 0.1233 | 3483 | 1014119 | put_skew_watch | 39.1715 |
| BTC | 2026-06-10 | 2.30 | 70.43 | 18.45 | 4.38 | 0.0763 | 1364 | 782934 | put_skew_watch | 36.0862 |
| ETH | 2026-06-09 | 1.30 | 86.95 | 17.62 | 3.15 | 0.0753 | 35634 | 229754 | put_skew_watch | 33.6825 |
| BTC | 2026-06-11 | 3.30 | 66.05 | 14.65 | 4.91 | 0.0651 | 277 | 266228 | put_skew_watch | 32.2088 |
| ETH | 2026-06-10 | 2.30 | 83.80 | 14.16 | 3.07 | 0.0807 | 9316 | 200801 | put_skew_watch | 29.4106 |
| ETH | 2026-06-26 | 18.30 | 65.17 | 5.70 | 5.48 | 0.0277 | 904523 | 535301 | put_skew_watch | 28.2897 |
| ETH | 2026-06-11 | 3.30 | 80.73 | 11.69 | 2.90 | 0.0465 | 2748 | 97359 | put_skew_watch | 25.8246 |
| BTC | 2026-06-19 | 11.30 | 51.39 | 7.92 | 3.65 | 0.0424 | 16210 | 1222013 | put_skew_watch | 25.4321 |
| BTC | 2026-06-26 | 18.30 | 47.74 | 6.11 | 3.69 | 0.0377 | 143668 | 3387148 | put_skew_watch | 25.1019 |
| ETH | 2026-06-19 | 11.30 | 68.49 | 6.52 | 3.32 | 0.0378 | 55466 | 180093 | put_skew_watch | 23.0840 |
| ETH | 2026-09-25 | 109.30 | 58.65 | 1.65 | -1.30 | 0.0154 | 293101 | 314572 | surface_context | 15.1840 |
| ETH | 2026-07-31 | 53.30 | 59.69 | 2.91 | 0.79 | 0.0213 | 139484 | 210279 | surface_context | 14.9147 |
| BTC | 2026-07-31 | 53.30 | 44.05 | 3.30 | 0.38 | 0.0220 | 37253 | 1652598 | surface_context | 14.8054 |
| BTC | 2026-09-25 | 109.30 | 43.80 | 1.68 | -0.84 | 0.0180 | 76390 | 1197202 | surface_context | 14.2851 |
| BTC | 2026-12-25 | 200.30 | 44.64 | 1.13 | -0.59 | 0.0207 | 80436 | 2123143 | surface_context | 13.5011 |
| ETH | 2026-12-25 | 200.30 | 59.95 | 1.14 | -0.78 | 0.0180 | 394740 | 108041 | surface_context | 13.2939 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
