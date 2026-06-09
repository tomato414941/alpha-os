# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BTC | 2026-06-09 | 0.30 | 44.37 | 17.66 | -6.32 | 0.1583 | 6719 | 3075447 | put_skew_watch | 40.2986 |
| ETH | 2026-06-09 | 0.30 | 66.95 | 22.62 | 0.24 | 0.4846 | 54280 | 416798 | put_skew_watch | 32.4853 |
| BTC | 2026-06-12 | 3.30 | 50.51 | 11.15 | 4.29 | 0.1075 | 31807 | 2047052 | put_skew_watch | 30.3286 |
| ETH | 2026-06-12 | 3.30 | 65.65 | 12.31 | 3.25 | 0.0876 | 147910 | 675185 | put_skew_watch | 29.6342 |
| ETH | 2026-06-10 | 1.30 | 66.71 | 17.85 | 0.44 | 0.1438 | 19630 | 262397 | put_skew_watch | 28.1543 |
| BTC | 2026-06-10 | 1.30 | 50.69 | 16.18 | -0.16 | 0.2055 | 4452 | 1202989 | put_skew_watch | 25.8178 |
| ETH | 2026-06-11 | 2.30 | 66.27 | 14.93 | 0.62 | 0.0896 | 10347 | 237367 | put_skew_watch | 25.3810 |
| ETH | 2026-06-26 | 17.30 | 60.18 | 4.59 | 3.01 | 0.0292 | 916238 | 1169406 | surface_context | 22.5816 |
| BTC | 2026-06-11 | 2.30 | 50.85 | 12.04 | 0.34 | 0.0824 | 3129 | 1071474 | put_skew_watch | 22.0809 |
| BTC | 2026-06-19 | 10.30 | 46.22 | 6.69 | 1.91 | 0.0434 | 17351 | 3694080 | put_skew_watch | 21.2301 |
| ETH | 2026-06-19 | 10.30 | 62.40 | 5.20 | 2.22 | 0.0403 | 62547 | 646822 | put_skew_watch | 20.1664 |
| BTC | 2026-06-26 | 17.30 | 44.31 | 5.62 | 1.10 | 0.0267 | 145229 | 5570531 | put_skew_watch | 19.6746 |
| ETH | 2026-09-25 | 108.30 | 56.76 | 1.51 | -2.17 | 0.0159 | 295098 | 863311 | surface_context | 17.2243 |
| BTC | 2026-09-25 | 108.30 | 42.59 | 1.80 | -1.86 | 0.0155 | 76718 | 4153551 | surface_context | 16.9922 |
| ETH | 2026-12-25 | 199.30 | 58.93 | 1.06 | -1.31 | 0.0188 | 396919 | 542787 | surface_context | 14.9758 |
| BTC | 2026-07-31 | 52.30 | 43.21 | 3.01 | 0.27 | 0.0189 | 39121 | 5586342 | surface_context | 14.8517 |
| ETH | 2026-07-31 | 52.30 | 57.17 | 2.30 | 0.64 | 0.0170 | 145260 | 1365415 | surface_context | 14.8434 |
| BTC | 2026-12-25 | 199.30 | 44.45 | 1.60 | -0.58 | 0.0128 | 80260 | 11154594 | surface_context | 14.6863 |
| BTC | 2026-08-28 | 80.30 | 42.94 | 2.11 | 0.35 | 0.0200 | 7184 | 2053739 | surface_context | 12.9390 |
| ETH | 2026-08-28 | 80.30 | 56.53 | 1.54 | -0.23 | 0.0170 | 18324 | 389167 | surface_context | 11.8192 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
