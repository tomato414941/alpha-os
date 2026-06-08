# Current Options Volatility Paper Tickets

This converts current Deribit IV-vs-realized and skew contexts into paper tickets. It is not a live options trade instruction.

| currency | expiry | structure | dte | atm iv | rv24 | prem24 | quote spread | max loss % | realized move % | prem/rv move | top depth USD | score | status | quote status | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| BTC | 2026-06-12 | none | 3.36 | 47.45 | 56.81 | -9.36 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 9.984041 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| ETH | 2026-06-12 | none | 3.36 | 62.67 | 71.03 | -8.36 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 6.053414 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-06-11 | none | 2.36 | 47.44 | 56.81 | -9.37 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 5.328579 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-06-10 | none | 1.36 | 45.83 | 56.81 | -10.98 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 4.635149 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| ETH | 2026-06-10 | none | 1.36 | 61.42 | 71.03 | -9.61 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 3.562126 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| ETH | 2026-06-11 | none | 2.36 | 62.85 | 71.03 | -8.18 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 3.213916 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-06-19 | none | 10.36 | 44.77 | 56.81 | -12.04 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 0.667900 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-06-26 | none | 17.36 | 43.13 | 56.81 | -13.68 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | -3.342100 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| ETH | 2026-06-19 | none | 10.36 | 59.66 | 71.03 | -11.37 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | -4.962223 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| ETH | 2026-06-26 | none | 17.36 | 58.29 | 71.03 | -12.74 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | -5.612301 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-06-09 | expiry_gamma_watch | 0.36 | 38.51 | 56.81 | -18.30 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | -7.062100 | too_close_to_expiry | quote_not_needed | expiry is too close for a clean paper ticket; no paper structure selected |
| BTC | 2027-03-26 | none | 290.36 | 44.71 | 56.81 | -12.10 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | -8.122100 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-07-31 | none | 52.36 | 42.42 | 56.81 | -14.39 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | -8.382100 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-12-25 | none | 199.36 | 44.19 | 56.81 | -12.62 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | -8.462100 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-09-25 | none | 108.36 | 42.40 | 56.81 | -14.41 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | -9.012100 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |

## Caveat

Long-vol tickets use a simple ATM straddle proxy from public Deribit quotes and top-of-book depth. They still lack multi-level sweep simulation, delta hedge PnL, margin, assignment/expiry handling, and realized-vol forecasts. Short premium candidates must be treated as capped-risk structures, not naked short options.
