# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BTC | 2026-06-12 | 3.88 | 56.08 | 16.69 | 7.24 | 0.0598 | 31680 | 1955221 | put_skew_watch | 41.8423 |
| ETH | 2026-06-12 | 3.88 | 72.20 | 16.70 | 5.15 | 0.0505 | 145548 | 765081 | put_skew_watch | 37.9458 |
| BTC | 2026-06-09 | 0.88 | 61.11 | 23.05 | 2.35 | 0.2292 | 4645 | 950673 | put_skew_watch | 36.9367 |
| ETH | 2026-06-09 | 0.88 | 71.70 | 24.71 | -0.20 | 0.1163 | 45706 | 310723 | put_skew_watch | 35.0298 |
| BTC | 2026-06-10 | 1.88 | 58.76 | 20.23 | 1.11 | 0.1087 | 1811 | 780786 | put_skew_watch | 31.3832 |
| ETH | 2026-06-10 | 1.88 | 71.90 | 21.07 | -0.44 | 0.0713 | 14246 | 237603 | put_skew_watch | 31.3371 |
| BTC | 2026-06-11 | 2.88 | 57.65 | 18.18 | 1.57 | 0.0827 | 569 | 290078 | put_skew_watch | 29.3731 |
| ETH | 2026-06-11 | 2.88 | 72.34 | 18.42 | 0.14 | 0.0754 | 6792 | 203920 | put_skew_watch | 27.6908 |
| ETH | 2026-06-26 | 17.88 | 63.74 | 5.77 | 4.93 | 0.0249 | 905262 | 708172 | put_skew_watch | 27.3872 |
| ETH | 2026-06-19 | 10.88 | 67.05 | 7.80 | 3.31 | 0.0355 | 59317 | 494810 | put_skew_watch | 24.8167 |
| BTC | 2026-06-19 | 10.88 | 48.84 | 9.13 | 2.68 | 0.0474 | 16363 | 1270698 | put_skew_watch | 24.7132 |
| BTC | 2026-06-26 | 17.88 | 46.16 | 7.32 | 2.54 | 0.0329 | 144539 | 4888193 | put_skew_watch | 24.1834 |
| BTC | 2026-09-25 | 108.88 | 42.96 | 1.98 | -1.54 | 0.0185 | 76282 | 2939231 | surface_context | 16.3737 |
| ETH | 2026-07-31 | 52.88 | 58.81 | 2.72 | 1.14 | 0.0171 | 143146 | 1036218 | surface_context | 16.1370 |
| ETH | 2026-09-25 | 108.88 | 57.80 | 1.71 | -1.61 | 0.0122 | 293646 | 553239 | surface_context | 16.1163 |
| ETH | 2026-12-25 | 199.88 | 59.41 | 1.17 | -1.35 | 0.0147 | 396872 | 328485 | surface_context | 14.9558 |
| BTC | 2026-07-31 | 52.88 | 43.62 | 3.81 | 0.19 | 0.0186 | 38085 | 1504486 | surface_context | 14.9109 |
| BTC | 2026-12-25 | 199.88 | 44.50 | 1.09 | -0.59 | 0.0134 | 80133 | 6127527 | surface_context | 13.9343 |
| BTC | 2026-08-28 | 80.88 | 43.43 | 2.93 | 0.47 | 0.0203 | 6853 | 647218 | surface_context | 13.4763 |
| BTC | 2027-03-26 | 290.88 | 45.09 | 0.91 |  | 0.0203 | 18367 | 2072471 | surface_context | 11.4499 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
