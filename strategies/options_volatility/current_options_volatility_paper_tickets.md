# Current Options Volatility Paper Tickets

This converts current Deribit IV-vs-realized and skew contexts into paper tickets. It is not a live options trade instruction.

| currency | expiry | structure | dte | atm iv | rv24 | prem24 | skew | term | volume USD | score | status | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| BTC | 2026-06-09 | short_put_spread | 1.41 | 80.07 | 48.08 | 31.99 | 23.81 | 4.36 | 919756 | 81.078056 | paper_short_put_spread_candidate | put skew and IV premium are rich versus recent realized vol |
| BTC | 2026-06-10 | short_put_spread | 2.41 | 75.71 | 48.08 | 27.63 | 21.59 | 4.54 | 607132 | 74.365432 | paper_short_put_spread_candidate | put skew and IV premium are rich versus recent realized vol |
| BTC | 2026-06-12 | calendar_spread | 4.41 | 67.70 | 48.08 | 19.62 | 14.77 | 12.30 | 1158560 | 57.846860 | paper_calendar_spread_watch | front IV premium and term spread are elevated |
| ETH | 2026-06-12 | calendar_spread | 4.41 | 81.70 | 65.34 | 16.36 | 10.94 | 11.15 | 330201 | 48.780101 | paper_calendar_spread_watch | front IV premium and term spread are elevated |
| ETH | 2026-06-08 | expiry_gamma_watch | 0.41 | 87.79 | 65.34 | 22.45 | 30.62 | -2.28 | 288229 | 48.358129 | too_close_to_expiry | expiry is too close for a clean paper ticket |
| ETH | 2026-06-09 | short_put_spread | 1.41 | 90.07 | 65.34 | 24.73 | 17.36 | 3.49 | 142077 | 47.721977 | too_thin | put skew is rich but option volume is thin |
| BTC | 2026-06-08 | expiry_gamma_watch | 0.41 | 67.77 | 48.08 | 19.69 | 31.73 | -12.30 | 964480 | 47.382780 | too_close_to_expiry | expiry is too close for a clean paper ticket |
| BTC | 2026-06-11 | short_put_spread | 3.41 | 71.17 | 48.08 | 23.09 | 14.78 | 3.47 | 164354 | 43.502654 | too_thin | put skew is rich but option volume is thin |
| ETH | 2026-06-10 | short_put_spread | 2.41 | 86.58 | 65.34 | 21.24 | 13.26 | 2.87 | 88143 | 39.458043 | too_thin | put skew is rich but option volume is thin |
| ETH | 2026-06-11 | none | 3.41 | 83.71 | 65.34 | 18.37 | 11.12 | 2.01 | 43613 | 31.543513 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-19 | none | 11.41 | 55.40 | 48.08 | 7.32 | 8.60 | 4.73 | 838443 | 21.486743 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-26 | none | 18.41 | 50.67 | 48.08 | 2.59 | 6.50 | 4.89 | 2802800 | 16.781100 | context_only | surface context exists but no paper structure is selected |
| ETH | 2026-06-19 | none | 11.41 | 70.55 | 65.34 | 5.21 | 6.42 | 3.82 | 83872 | 15.533772 | context_only | surface context exists but no paper structure is selected |
| ETH | 2026-06-26 | none | 18.41 | 66.73 | 65.34 | 1.39 | 5.04 | 6.15 | 142612 | 12.722512 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-07-31 | none | 53.41 | 45.78 | 48.08 | -2.30 | 3.46 | 0.95 | 1367235 | 3.475535 | context_only | surface context exists but no paper structure is selected |

## Caveat

These tickets still lack option spread quotes, delta hedge PnL, margin, assignment/expiry handling, and realized-vol forecasts. Short premium candidates must be treated as capped-risk structures, not naked short options.
