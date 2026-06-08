# Current Options Volatility Paper Tickets

This converts current Deribit IV-vs-realized and skew contexts into paper tickets. It is not a live options trade instruction.

| currency | expiry | structure | dte | atm iv | rv24 | prem24 | skew | term | volume USD | score | status | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| BTC | 2026-06-09 | none | 1.30 | 73.46 | 81.84 | -8.38 | 23.81 | 3.03 | 1014119 | 19.476319 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-10 | none | 2.30 | 70.43 | 81.84 | -11.41 | 18.45 | 4.38 | 782934 | 12.205134 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-11 | none | 3.30 | 66.05 | 81.84 | -15.79 | 14.65 | 4.91 | 266228 | 4.038428 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-12 | none | 4.30 | 61.14 | 81.84 | -20.70 | 12.68 | 9.75 | 2061821 | 3.794021 | context_only | surface context exists but no paper structure is selected |
| ETH | 2026-06-09 | none | 1.30 | 86.95 | 113.14 | -26.19 | 17.62 | 3.15 | 229754 | -5.194046 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-08 | expiry_gamma_watch | 0.30 | 64.34 | 81.84 | -17.50 | 14.96 | -9.12 | 1129768 | -6.408032 | too_close_to_expiry | expiry is too close for a clean paper ticket |
| ETH | 2026-06-10 | none | 2.30 | 83.80 | 113.14 | -29.34 | 14.16 | 3.07 | 200801 | -11.912999 | context_only | surface context exists but no paper structure is selected |
| ETH | 2026-06-12 | none | 4.30 | 77.83 | 113.14 | -35.31 | 9.87 | 9.34 | 550491 | -15.553309 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-19 | none | 11.30 | 51.39 | 81.84 | -30.45 | 7.92 | 3.65 | 1222013 | -17.655787 | context_only | surface context exists but no paper structure is selected |
| ETH | 2026-06-11 | none | 3.30 | 80.73 | 113.14 | -32.41 | 11.69 | 2.90 | 97359 | -17.726441 | context_only | surface context exists but no paper structure is selected |
| ETH | 2026-06-08 | expiry_gamma_watch | 0.30 | 81.27 | 113.14 | -31.87 | 18.75 | -5.68 | 379550 | -17.744250 | too_close_to_expiry | expiry is too close for a clean paper ticket |
| BTC | 2026-06-26 | none | 18.30 | 47.74 | 81.84 | -34.10 | 6.11 | 3.69 | 3387148 | -21.297800 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-07-31 | none | 53.30 | 44.05 | 81.84 | -37.79 | 3.30 | 0.38 | 1652598 | -32.455202 | context_only | surface context exists but no paper structure is selected |
| BTC | 2027-03-26 | none | 291.30 | 45.23 | 81.84 | -36.61 | 0.95 | 0.00 | 2413391 | -33.244409 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-12-25 | none | 200.30 | 44.64 | 81.84 | -37.20 | 1.13 | -0.59 | 2123143 | -33.944657 | context_only | surface context exists but no paper structure is selected |

## Caveat

These tickets still lack option spread quotes, delta hedge PnL, margin, assignment/expiry handling, and realized-vol forecasts. Short premium candidates must be treated as capped-risk structures, not naked short options.
