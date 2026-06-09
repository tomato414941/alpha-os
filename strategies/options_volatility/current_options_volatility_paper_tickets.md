# Current Options Volatility Paper Tickets

This converts current Deribit IV-vs-realized and skew contexts into paper tickets. It is not a live options trade instruction.

| currency | expiry | structure | dte | atm iv | rv24 | prem24 | quote spread | max loss % | realized move % | prem/rv move | top depth USD | score | status | quote status | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| ETH | 2026-06-10 | none | 1.30 | 66.71 | 72.32 | -5.61 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 12.943097 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| ETH | 2026-06-09 | expiry_gamma_watch | 0.30 | 66.95 | 72.32 | -5.37 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 12.907498 | too_close_to_expiry | quote_not_needed | expiry is too close for a clean paper ticket; no paper structure selected |
| BTC | 2026-06-10 | none | 1.30 | 50.69 | 57.42 | -6.73 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 10.653589 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-06-12 | none | 3.30 | 50.51 | 57.42 | -6.91 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 10.577652 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| ETH | 2026-06-11 | none | 2.30 | 66.27 | 72.32 | -6.05 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 9.738067 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| ETH | 2026-06-12 | none | 3.30 | 65.65 | 72.32 | -6.67 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 9.565885 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-06-11 | none | 2.30 | 50.85 | 57.42 | -6.57 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 6.882074 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-06-09 | expiry_gamma_watch | 0.30 | 44.37 | 57.42 | -13.05 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 2.610600 | too_close_to_expiry | quote_not_needed | expiry is too close for a clean paper ticket; no paper structure selected |
| BTC | 2026-06-19 | none | 10.30 | 46.22 | 57.42 | -11.20 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | 0.400600 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| ETH | 2026-06-19 | none | 10.30 | 62.40 | 72.32 | -9.92 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | -1.852478 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| ETH | 2026-06-26 | none | 17.30 | 60.18 | 72.32 | -12.14 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | -3.369894 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-06-26 | none | 17.30 | 44.31 | 57.42 | -13.11 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | -3.389400 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-07-31 | none | 52.30 | 43.21 | 57.42 | -14.21 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | -7.929400 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-12-25 | none | 199.30 | 44.45 | 57.42 | -12.97 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | -8.369400 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2027-03-26 | none | 290.30 | 45.03 | 57.42 | -12.39 | 0.0000 | 0.00 | 0.00 | 0.00 | 0 | -8.499400 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |

## Caveat

Long-vol tickets use a simple ATM straddle proxy from public Deribit quotes and top-of-book depth. They still lack multi-level sweep simulation, delta hedge PnL, margin, assignment/expiry handling, and realized-vol forecasts. Short premium candidates must be treated as capped-risk structures, not naked short options.
