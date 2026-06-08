# Current Options Volatility Paper Tickets

This converts current Deribit IV-vs-realized and skew contexts into paper tickets. It is not a live options trade instruction.

| currency | expiry | structure | dte | atm iv | rv24 | prem24 | skew | term | volume USD | score | status | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ETH | 2026-06-26 | long_vol_spread | 18.28 | 64.92 | 111.17 | -46.25 | 4.77 | 5.42 | 543296 | 72.982296 | paper_long_vol_candidate | IV is cheap versus recent realized vol; test capped-premium long-vol structure |
| ETH | 2026-07-31 | long_vol_spread | 53.28 | 59.50 | 111.17 | -51.67 | 2.30 | 0.74 | 286603 | 70.995603 | paper_long_vol_candidate | IV is cheap versus recent realized vol; test capped-premium long-vol structure |
| ETH | 2026-09-25 | long_vol_spread | 109.28 | 58.61 | 111.17 | -52.56 | 1.60 | -1.31 | 368904 | 70.527904 | paper_long_vol_candidate | IV is cheap versus recent realized vol; test capped-premium long-vol structure |
| ETH | 2026-06-12 | long_vol_spread | 4.28 | 79.01 | 111.17 | -32.16 | 11.01 | 10.63 | 548808 | 70.347808 | paper_long_vol_candidate | IV is cheap versus recent realized vol; test capped-premium long-vol structure |
| ETH | 2027-03-26 | long_vol_spread | 291.28 | 61.08 | 111.17 | -50.09 | 0.84 | 0.00 | 257858 | 67.186858 | paper_long_vol_candidate | IV is cheap versus recent realized vol; test capped-premium long-vol structure |
| BTC | 2026-06-26 | long_vol_spread | 18.28 | 48.22 | 81.65 | -33.43 | 6.18 | 3.80 | 3492665 | 62.412300 | paper_long_vol_candidate | IV is cheap versus recent realized vol; test capped-premium long-vol structure |
| BTC | 2026-06-19 | long_vol_spread | 11.28 | 51.98 | 81.65 | -29.67 | 8.18 | 3.76 | 1231162 | 58.843462 | paper_long_vol_candidate | IV is cheap versus recent realized vol; test capped-premium long-vol structure |
| BTC | 2026-07-31 | long_vol_spread | 53.28 | 44.42 | 81.65 | -37.23 | 3.41 | 0.42 | 1591606 | 58.653906 | paper_long_vol_candidate | IV is cheap versus recent realized vol; test capped-premium long-vol structure |
| BTC | 2026-08-28 | long_vol_spread | 81.28 | 44.00 | 81.65 | -37.65 | 2.71 | 0.22 | 363346 | 56.945646 | paper_long_vol_candidate | IV is cheap versus recent realized vol; test capped-premium long-vol structure |
| BTC | 2026-09-25 | long_vol_spread | 109.28 | 43.78 | 81.65 | -37.87 | 1.85 | -1.10 | 1196667 | 56.918967 | paper_long_vol_candidate | IV is cheap versus recent realized vol; test capped-premium long-vol structure |
| BTC | 2026-12-25 | long_vol_spread | 200.28 | 44.88 | 81.65 | -36.77 | 1.11 | -0.52 | 2091087 | 55.973387 | paper_long_vol_candidate | IV is cheap versus recent realized vol; test capped-premium long-vol structure |
| BTC | 2027-03-26 | long_vol_spread | 291.28 | 45.40 | 81.65 | -36.25 | 0.89 | 0.00 | 2414443 | 55.556743 | paper_long_vol_candidate | IV is cheap versus recent realized vol; test capped-premium long-vol structure |
| BTC | 2026-06-09 | none | 1.28 | 74.66 | 81.65 | -6.99 | 23.92 | 3.73 | 990017 | 21.647717 | context_only | surface context exists but no paper structure is selected |
| BTC | 2026-06-08 | expiry_gamma_watch | 0.28 | 68.60 | 81.65 | -13.05 | 36.21 | -6.06 | 1128156 | 19.285856 | too_close_to_expiry | expiry is too close for a clean paper ticket |
| BTC | 2026-06-10 | none | 2.28 | 70.93 | 81.65 | -10.72 | 18.98 | 4.89 | 776393 | 13.924093 | context_only | surface context exists but no paper structure is selected |

## Caveat

These tickets still lack option spread quotes, delta hedge PnL, margin, assignment/expiry handling, and realized-vol forecasts. Short premium candidates must be treated as capped-risk structures, not naked short options.
