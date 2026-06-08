# Current Options Volatility Paper Tickets

This converts current Deribit IV-vs-realized and skew contexts into paper tickets. It is not a live options trade instruction.

| currency | expiry | structure | dte | atm iv | rv24 | prem24 | quote spread | premium % | premium USD | score | status | quote status | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| ETH | 2026-06-26 | long_vol_spread | 18.28 | 64.92 | 111.17 | -46.25 | 0.0254 | 11.95 | 200 | 76.982296 | paper_long_vol_quote_candidate | quote_executable_watch | IV is cheap versus recent realized vol; test capped-premium long-vol structure; ATM option pair quote is present with acceptable spread |
| ETH | 2026-06-12 | long_vol_spread | 4.28 | 79.01 | 111.17 | -32.16 | 0.0438 | 7.00 | 117 | 74.347808 | paper_long_vol_quote_candidate | quote_executable_watch | IV is cheap versus recent realized vol; test capped-premium long-vol structure; ATM option pair quote is present with acceptable spread |
| BTC | 2026-06-26 | long_vol_spread | 18.28 | 48.22 | 81.65 | -33.43 | 0.0233 | 8.70 | 5464 | 66.412300 | paper_long_vol_quote_candidate | quote_executable_watch | IV is cheap versus recent realized vol; test capped-premium long-vol structure; ATM option pair quote is present with acceptable spread |
| BTC | 2026-06-19 | long_vol_spread | 11.28 | 51.98 | 81.65 | -29.67 | 0.0205 | 7.40 | 4645 | 62.843462 | paper_long_vol_quote_candidate | quote_executable_watch | IV is cheap versus recent realized vol; test capped-premium long-vol structure; ATM option pair quote is present with acceptable spread |
| ETH | 2026-07-31 | long_vol_spread | 53.28 | 59.50 | 111.17 | -51.67 | 0.0109 | 18.40 | 308 | 58.995603 | paper_long_vol_quote_blocked | premium_too_large | IV is cheap versus recent realized vol; test capped-premium long-vol structure; ATM long-vol proxy premium is too large relative to notional |
| ETH | 2026-09-25 | long_vol_spread | 109.28 | 58.61 | 111.17 | -52.56 | 0.0137 | 25.75 | 434 | 58.527904 | paper_long_vol_quote_blocked | premium_too_large | IV is cheap versus recent realized vol; test capped-premium long-vol structure; ATM long-vol proxy premium is too large relative to notional |
| ETH | 2027-03-26 | long_vol_spread | 291.28 | 61.08 | 111.17 | -50.09 | 0.0175 | 43.15 | 742 | 55.186858 | paper_long_vol_quote_blocked | premium_too_large | IV is cheap versus recent realized vol; test capped-premium long-vol structure; ATM long-vol proxy premium is too large relative to notional |
| BTC | 2026-07-31 | long_vol_spread | 53.28 | 44.42 | 81.65 | -37.23 | 0.0148 | 13.60 | 8565 | 46.653906 | paper_long_vol_quote_blocked | premium_too_large | IV is cheap versus recent realized vol; test capped-premium long-vol structure; ATM long-vol proxy premium is too large relative to notional |
| BTC | 2026-08-28 | long_vol_spread | 81.28 | 44.00 | 81.65 | -37.65 | 0.0182 | 16.65 | 10513 | 44.945646 | paper_long_vol_quote_blocked | premium_too_large | IV is cheap versus recent realized vol; test capped-premium long-vol structure; ATM long-vol proxy premium is too large relative to notional |
| BTC | 2026-09-25 | long_vol_spread | 109.28 | 43.78 | 81.65 | -37.87 | 0.0157 | 19.30 | 12228 | 44.918967 | paper_long_vol_quote_blocked | premium_too_large | IV is cheap versus recent realized vol; test capped-premium long-vol structure; ATM long-vol proxy premium is too large relative to notional |
| BTC | 2026-12-25 | long_vol_spread | 200.28 | 44.88 | 81.65 | -36.77 | 0.0189 | 26.65 | 17063 | 43.973387 | paper_long_vol_quote_blocked | premium_too_large | IV is cheap versus recent realized vol; test capped-premium long-vol structure; ATM long-vol proxy premium is too large relative to notional |
| BTC | 2027-03-26 | long_vol_spread | 291.28 | 45.40 | 81.65 | -36.25 | 0.0218 | 32.50 | 21030 | 43.556743 | paper_long_vol_quote_blocked | premium_too_large | IV is cheap versus recent realized vol; test capped-premium long-vol structure; ATM long-vol proxy premium is too large relative to notional |
| BTC | 2026-06-09 | none | 1.28 | 74.66 | 81.65 | -6.99 | 0.0000 | 0.00 | 0 | 21.647717 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |
| BTC | 2026-06-08 | expiry_gamma_watch | 0.28 | 68.60 | 81.65 | -13.05 | 0.0000 | 0.00 | 0 | 19.285856 | too_close_to_expiry | quote_not_needed | expiry is too close for a clean paper ticket; no paper structure selected |
| BTC | 2026-06-10 | none | 2.28 | 70.93 | 81.65 | -10.72 | 0.0000 | 0.00 | 0 | 13.924093 | context_only | quote_not_needed | surface context exists but no paper structure is selected; no paper structure selected |

## Caveat

These tickets use only a simple ATM call+put quote proxy. They still lack full option spread construction, delta hedge PnL, margin, assignment/expiry handling, and realized-vol forecasts. Short premium candidates must be treated as capped-risk structures, not naked short options.
