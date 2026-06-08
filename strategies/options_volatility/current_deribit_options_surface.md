# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BTC | 2026-06-09 | 0.36 | 38.51 | 13.24 | -7.32 | 0.1459 | 6706 | 3031823 | put_skew_watch | 37.8965 |
| BTC | 2026-06-12 | 3.36 | 47.45 | 14.22 | 2.68 | 0.0935 | 31769 | 2446141 | put_skew_watch | 30.2835 |
| ETH | 2026-06-12 | 3.36 | 62.67 | 10.68 | 3.01 | 0.0709 | 148181 | 721314 | put_skew_watch | 27.5870 |
| BTC | 2026-06-10 | 1.36 | 45.83 | 14.36 | -1.61 | 0.2639 | 4358 | 1257249 | put_skew_watch | 26.7909 |
| ETH | 2026-06-10 | 1.36 | 61.42 | 12.89 | -1.43 | 0.1315 | 17163 | 280026 | put_skew_watch | 25.1688 |
| BTC | 2026-06-11 | 2.36 | 47.44 | 13.61 | -0.01 | 0.1342 | 3048 | 1090679 | put_skew_watch | 22.8835 |
| BTC | 2026-06-19 | 10.36 | 44.77 | 8.07 | 1.64 | 0.0397 | 17324 | 3886533 | put_skew_watch | 22.0988 |
| ETH | 2026-06-11 | 2.36 | 62.85 | 10.98 | 0.18 | 0.0804 | 9178 | 231816 | put_skew_watch | 20.5071 |
| ETH | 2026-06-26 | 17.36 | 58.29 | 3.85 | 2.09 | 0.0281 | 916444 | 1185599 | surface_context | 20.0098 |
| BTC | 2026-06-26 | 17.36 | 43.13 | 6.63 | 0.71 | 0.0429 | 145244 | 5532106 | put_skew_watch | 19.8691 |
| ETH | 2026-06-09 | 0.36 | 59.61 | 5.09 | -1.81 | 0.3326 | 52630 | 414184 | put_skew_watch | 18.3832 |
| ETH | 2026-06-19 | 10.36 | 59.66 | 4.43 | 1.37 | 0.0393 | 61829 | 605677 | surface_context | 17.6649 |
| BTC | 2026-09-25 | 108.36 | 42.40 | 2.40 | -1.79 | 0.0198 | 76721 | 4132149 | surface_context | 17.4415 |
| ETH | 2026-09-25 | 108.36 | 56.75 | 1.65 | -2.01 | 0.0150 | 295211 | 865873 | surface_context | 17.0475 |
| BTC | 2026-07-31 | 52.36 | 42.42 | 3.00 | 0.01 | 0.0173 | 39106 | 5450505 | surface_context | 14.3140 |
| BTC | 2026-12-25 | 199.36 | 44.19 | 1.16 | -0.52 | 0.0115 | 80607 | 10334138 | surface_context | 14.0976 |
| ETH | 2026-07-31 | 52.36 | 56.20 | 2.05 | -0.21 | 0.0157 | 144955 | 1291510 | surface_context | 13.7110 |
| ETH | 2026-12-25 | 199.36 | 58.76 | 0.55 | -0.91 | 0.0152 | 396915 | 528345 | surface_context | 13.6612 |
| BTC | 2026-08-28 | 80.36 | 42.41 | 2.16 | 0.01 | 0.0187 | 7175 | 2016724 | surface_context | 12.3032 |
| ETH | 2026-08-28 | 80.36 | 56.41 | 1.50 | -0.34 | 0.0172 | 18302 | 380329 | surface_context | 11.9884 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
