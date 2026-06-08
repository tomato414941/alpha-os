# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BTC | 2026-06-08 | 0.33 | 65.15 | 26.60 | -12.28 | 0.2550 | 5300 | 1106823 | put_skew_watch | 60.4185 |
| ETH | 2026-06-08 | 0.33 | 84.65 | 30.50 | -4.11 | 0.2894 | 54966 | 376532 | put_skew_watch | 48.4571 |
| BTC | 2026-06-12 | 4.33 | 64.91 | 14.92 | 11.43 | 0.0627 | 30990 | 1773907 | put_skew_watch | 48.3947 |
| ETH | 2026-06-12 | 4.33 | 79.59 | 10.17 | 10.98 | 0.0510 | 139863 | 510018 | front_vol_premium_watch | 42.8713 |
| BTC | 2026-06-09 | 1.33 | 77.43 | 25.69 | 3.78 | 0.0798 | 3452 | 1000005 | put_skew_watch | 42.6286 |
| BTC | 2026-06-10 | 2.33 | 73.65 | 20.56 | 4.84 | 0.0733 | 1326 | 742508 | put_skew_watch | 39.0869 |
| ETH | 2026-06-09 | 1.33 | 88.76 | 20.58 | 3.42 | 0.0740 | 35191 | 219704 | put_skew_watch | 37.1603 |
| BTC | 2026-06-11 | 3.33 | 68.81 | 15.75 | 3.90 | 0.0764 | 251 | 251454 | put_skew_watch | 31.1986 |
| ETH | 2026-06-10 | 2.33 | 85.34 | 15.03 | 3.43 | 0.0828 | 9317 | 169359 | put_skew_watch | 30.9225 |
| BTC | 2026-06-19 | 11.33 | 53.48 | 9.06 | 4.21 | 0.0397 | 15913 | 1011303 | put_skew_watch | 27.6072 |
| BTC | 2026-06-26 | 18.33 | 49.27 | 6.76 | 4.47 | 0.0396 | 143670 | 3357892 | put_skew_watch | 27.3042 |
| ETH | 2026-06-26 | 18.33 | 65.14 | 4.81 | 5.30 | 0.0265 | 904874 | 539042 | front_vol_premium_watch | 27.0453 |
| ETH | 2026-06-11 | 3.33 | 81.91 | 12.48 | 2.32 | 0.0652 | 2572 | 91446 | put_skew_watch | 25.3712 |
| ETH | 2026-06-19 | 11.33 | 68.61 | 6.17 | 3.47 | 0.0341 | 55378 | 174579 | put_skew_watch | 23.0271 |
| BTC | 2026-07-31 | 53.33 | 44.80 | 3.97 | 0.98 | 0.0251 | 37247 | 1640924 | surface_context | 16.6659 |
| ETH | 2026-09-25 | 109.33 | 58.90 | 1.67 | -1.30 | 0.0165 | 293177 | 242802 | surface_context | 15.0895 |
| BTC | 2026-09-25 | 109.33 | 43.94 | 1.93 | -0.98 | 0.0211 | 76394 | 1169403 | surface_context | 14.7989 |
| ETH | 2026-07-31 | 53.33 | 59.84 | 2.38 | 0.82 | 0.0174 | 139451 | 201269 | surface_context | 14.4335 |
| ETH | 2026-12-25 | 200.33 | 60.20 | 1.16 | -1.12 | 0.0169 | 394800 | 108214 | surface_context | 13.9969 |
| BTC | 2026-12-25 | 200.33 | 44.92 | 1.14 | -0.52 | 0.0197 | 80440 | 2096989 | surface_context | 13.3677 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
