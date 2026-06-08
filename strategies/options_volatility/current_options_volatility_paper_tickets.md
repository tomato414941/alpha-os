# Current Options Volatility Paper Tickets

This converts current Deribit IV-vs-realized and skew contexts into paper tickets. It is not a live options trade instruction.

| currency | expiry | structure | dte | atm iv | rv24 | prem24 | skew | term | volume USD | score | status | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| BTC | 2026-06-09 | none | 1.33 | 77.43 | 81.41 | -3.98 | 25.69 | 3.78 | 1000005 | 26.494405 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-10 | none | 2.33 | 73.65 | 81.41 | -7.76 | 20.56 | 4.84 | 742508 | 18.386908 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-12 | none | 4.33 | 64.91 | 81.41 | -16.50 | 14.92 | 11.43 | 1773907 | 11.628307 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-11 | none | 3.33 | 68.81 | 81.41 | -12.60 | 15.75 | 3.90 | 251454 | 7.305854 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-08 | expiry_gamma_watch | 0.33 | 65.15 | 81.41 | -16.26 | 26.60 | -12.28 | 1106823 | 6.451223 | too_close_to_expiry | expiry is too close for a clean paper ticket |
| ETH | 2026-06-09 | none | 1.33 | 88.76 | 112.34 | -23.58 | 20.58 | 3.42 | 219704 | 0.637704 | context_only | surface context exists but no paper structure is selected |
| ETH | 2026-06-08 | expiry_gamma_watch | 0.33 | 84.65 | 112.34 | -27.69 | 30.50 | -4.11 | 376532 | -1.815468 | too_close_to_expiry | expiry is too close for a clean paper ticket |
| ETH | 2026-06-10 | none | 2.33 | 85.34 | 112.34 | -27.00 | 15.03 | 3.43 | 169359 | -8.372641 | context_only | surface context exists but no paper structure is selected |
| ETH | 2026-06-12 | none | 4.33 | 79.58 | 112.34 | -32.76 | 10.17 | 10.97 | 510018 | -11.101982 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-19 | none | 11.33 | 53.48 | 81.41 | -27.93 | 9.06 | 4.21 | 1011303 | -13.644297 | context_only | surface context exists but no paper structure is selected |
| ETH | 2026-06-11 | none | 3.33 | 81.91 | 112.34 | -30.43 | 12.48 | 2.33 | 91446 | -15.535554 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-26 | none | 18.33 | 49.27 | 81.41 | -32.14 | 6.76 | 4.47 | 3357892 | -17.905600 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-07-31 | none | 53.33 | 44.80 | 81.41 | -36.61 | 3.97 | 0.98 | 1640924 | -30.014676 | context_only | surface context exists but no paper structure is selected |
| BTC | 2027-03-26 | none | 291.33 | 45.44 | 81.41 | -35.97 | 0.97 | 0.00 | 2409499 | -32.586101 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-12-25 | none | 200.33 | 44.92 | 81.41 | -36.49 | 1.14 | -0.52 | 2096989 | -33.248611 | context_only | surface context exists but no paper structure is selected |

## Caveat

These tickets still lack option spread quotes, delta hedge PnL, margin, assignment/expiry handling, and realized-vol forecasts. Short premium candidates must be treated as capped-risk structures, not naked short options.
