# Current Options Volatility Paper Tickets

This converts current Deribit IV-vs-realized and skew contexts into paper tickets. It is not a live options trade instruction.

| currency | expiry | structure | dte | atm iv | rv24 | prem24 | skew | term | volume USD | score | status | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| BTC | 2026-06-09 | short_put_spread | 1.49 | 83.30 | 47.87 | 35.43 | 23.22 | 5.49 | 981392 | 85.117692 | paper_short_put_spread_candidate | put skew and IV premium are rich versus recent realized vol |
| BTC | 2026-06-10 | short_put_spread | 2.49 | 77.81 | 47.87 | 29.94 | 21.50 | 5.01 | 557453 | 77.003753 | paper_short_put_spread_candidate | put skew and IV premium are rich versus recent realized vol |
| BTC | 2026-06-12 | short_put_spread | 4.49 | 68.36 | 47.87 | 20.49 | 18.74 | 12.42 | 1189878 | 72.836178 | paper_short_put_spread_candidate | put skew and IV premium are rich versus recent realized vol |
| BTC | 2026-06-08 | expiry_gamma_watch | 0.49 | 75.87 | 47.87 | 28.00 | 32.77 | -7.43 | 959012 | 56.725312 | too_close_to_expiry | expiry is too close for a clean paper ticket |
| BTC | 2026-06-11 | short_put_spread | 3.49 | 72.80 | 47.87 | 24.93 | 18.38 | 4.44 | 153713 | 49.900013 | too_thin | put skew is rich but option volume is thin |
| ETH | 2026-06-08 | expiry_gamma_watch | 0.49 | 89.46 | 65.07 | 24.39 | 27.41 | -0.43 | 284011 | 47.083911 | too_close_to_expiry | expiry is too close for a clean paper ticket |
| ETH | 2026-06-09 | short_put_spread | 1.49 | 89.89 | 65.07 | 24.82 | 14.05 | 4.07 | 137371 | 45.077271 | too_thin | put skew is rich but option volume is thin |
| ETH | 2026-06-12 | calendar_spread | 4.49 | 81.25 | 65.07 | 16.18 | 9.65 | 8.45 | 340811 | 44.620711 | paper_calendar_spread_watch | front IV premium and term spread are elevated |
| ETH | 2026-06-10 | none | 2.49 | 85.82 | 65.07 | 20.75 | 11.62 | 2.57 | 80952 | 35.020852 | context_only | surface context exists but no paper structure is selected |
| ETH | 2026-06-11 | none | 3.49 | 83.25 | 65.07 | 18.18 | 9.61 | 2.00 | 39702 | 29.829602 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-19 | none | 11.49 | 55.94 | 47.87 | 8.07 | 10.76 | 4.71 | 872866 | 24.409166 | context_only | surface context exists but no paper structure is selected |
| ETH | 2026-06-19 | none | 11.49 | 72.80 | 65.07 | 7.73 | 6.42 | 4.50 | 99504 | 18.749404 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-26 | none | 18.49 | 51.23 | 47.87 | 3.36 | 6.43 | 5.12 | 2835496 | 17.741796 | context_only | surface context exists but no paper structure is selected |
| ETH | 2026-06-26 | none | 18.49 | 68.30 | 65.07 | 3.23 | 4.55 | 6.74 | 144610 | 14.664510 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-07-31 | none | 53.49 | 46.11 | 47.87 | -1.76 | 3.37 | 1.16 | 1370647 | 4.136947 | context_only | surface context exists but no paper structure is selected |

## Caveat

These tickets still lack option spread quotes, delta hedge PnL, margin, assignment/expiry handling, and realized-vol forecasts. Short premium candidates must be treated as capped-risk structures, not naked short options.
