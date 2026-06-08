# Protocol Fee Price-Lag Forward Labels

This labels stored fee-growth price-lag observations. Positive directional return means the observation's direction was right before fees, funding PnL, and slippage.

- total rows: `34`
- labeled 4h rows: `18`

| observed at | token | status | dir | priority | dir 4h | dir 12h | dir 24h | dir 7d | label status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-08T09:52:00+00:00 | MORPHO | fee_growth_price_lag_candidate | 1 | 58.5616 | 0.08341800 | 0.03854877 |  |  | labeled_12h_pending_24h |
| 2026-06-08T09:52:00+00:00 | JUP | fee_growth_price_lag_candidate | 1 | 93.9263 | 0.01957319 | 0.02055390 |  |  | labeled_12h_pending_24h |
| 2026-06-08T09:52:00+00:00 | CRV | fee_growth_price_lag_candidate | 1 | 59.9474 | 0.01677241 | 0.01876309 |  |  | labeled_12h_pending_24h |
| 2026-06-08T00:46:00+00:00 | HYPE | fee_decay_price_weakness_context | -1 | 11.2882 | 0.01234720 | -0.01578578 |  |  | labeled_12h_pending_24h |
| 2026-06-07T22:36:00+00:00 | CRV | fee_growth_price_lag_candidate | 1 | 59.1753 | 0.00396679 | 0.01829538 | 0.02064387 |  | labeled_24h_pending_7d |
| 2026-06-08T09:52:00+00:00 | AAVE | fee_growth_price_lag_candidate | 1 | 87.9800 | 0.00083404 | 0.00807536 |  |  | labeled_12h_pending_24h |
| 2026-06-08T00:41:00+00:00 | CRV | fee_growth_price_lag_candidate | 1 | 59.1753 | -0.00023355 | 0.02806976 |  |  | labeled_12h_pending_24h |
| 2026-06-07T22:36:00+00:00 | MORPHO | fee_growth_price_lag_candidate | 1 | 60.8602 | -0.00048924 | -0.00257600 | 0.04955146 |  | labeled_24h_pending_7d |
| 2026-06-07T22:36:00+00:00 | JUP | fee_growth_price_lag_candidate | 1 | 94.3453 | -0.00369332 | 0.00112637 | 0.01792571 |  | labeled_24h_pending_7d |
| 2026-06-08T00:46:00+00:00 | CRV | fee_growth_price_lag_candidate | 1 | 59.9474 | -0.00697578 | 0.02113666 |  |  | labeled_12h_pending_24h |
| 2026-06-08T00:41:00+00:00 | MORPHO | fee_growth_price_lag_candidate | 1 | 60.8602 | -0.01139318 | 0.08218454 |  |  | labeled_12h_pending_24h |
| 2026-06-08T00:41:00+00:00 | JUP | fee_growth_price_lag_candidate | 1 | 94.3453 | -0.01405048 | 0.03190270 |  |  | labeled_12h_pending_24h |
| 2026-06-07T22:36:00+00:00 | AAVE | fee_growth_price_lag_candidate | 1 | 87.8950 | -0.01479578 | -0.01203360 | -0.00198273 |  | labeled_24h_pending_7d |
| 2026-06-08T00:41:00+00:00 | AAVE | fee_growth_price_lag_candidate | 1 | 87.8950 | -0.02321015 | 0.00577144 |  |  | labeled_12h_pending_24h |
| 2026-06-08T00:46:00+00:00 | AAVE | fee_growth_price_lag_candidate | 1 | 87.9800 | -0.02549486 | 0.00341894 |  |  | labeled_12h_pending_24h |
| 2026-06-08T00:46:00+00:00 | JUP | fee_growth_price_lag_candidate | 1 | 93.9263 | -0.02556743 | 0.01984896 |  |  | labeled_12h_pending_24h |
| 2026-06-08T00:46:00+00:00 | MORPHO | fee_growth_price_lag_candidate | 1 | 58.5616 | -0.02834072 | 0.06363280 |  |  | labeled_12h_pending_24h |
| 2026-06-08T09:52:00+00:00 | HYPE | fee_decay_price_weakness_context | -1 | 11.2882 | -0.04442721 | -0.04236810 |  |  | labeled_12h_pending_24h |
| 2026-06-07T22:36:00+00:00 | UNI | fee_growth_price_lag_candidate | 1 | 77.9108 |  |  |  |  | pending_4h |
| 2026-06-08T00:41:00+00:00 | UNI | fee_growth_price_lag_candidate | 1 | 77.9108 |  |  |  |  | pending_4h |
| 2026-06-08T00:46:00+00:00 | UNI | fee_growth_price_lag_candidate | 1 | 77.8233 |  |  |  |  | pending_4h |
| 2026-06-08T09:52:00+00:00 | UNI | fee_growth_price_lag_candidate | 1 | 77.8233 |  |  |  |  | pending_4h |
| 2026-06-08T00:46:00+00:00 | PENDLE | fee_growth_price_lag_candidate | 1 | 40.2142 |  |  |  |  | pending_4h |
| 2026-06-08T09:52:00+00:00 | PENDLE | fee_growth_price_lag_candidate | 1 | 40.2142 |  |  |  |  | pending_4h |
| 2026-06-07T22:36:00+00:00 | PENDLE | fee_growth_price_lag_candidate | 1 | 39.2389 |  |  |  |  | pending_4h |
| 2026-06-08T00:41:00+00:00 | PENDLE | fee_growth_price_lag_candidate | 1 | 39.2389 |  |  |  |  | pending_4h |
| 2026-06-07T22:36:00+00:00 | UNI | fee_decay_price_weakness_context | -1 | 32.1903 |  |  |  |  | pending_4h |
| 2026-06-08T00:41:00+00:00 | UNI | fee_decay_price_weakness_context | -1 | 32.1903 |  |  |  |  | pending_4h |
| 2026-06-08T00:46:00+00:00 | UNI | fee_decay_price_weakness_context | -1 | 31.9772 |  |  |  |  | pending_4h |
| 2026-06-08T09:52:00+00:00 | UNI | fee_decay_price_weakness_context | -1 | 31.9772 |  |  |  |  | pending_4h |
| 2026-06-07T22:36:00+00:00 | SOL | fee_decay_price_weakness_context | -1 | 10.7636 |  |  |  |  | pending_4h |
| 2026-06-08T00:41:00+00:00 | SOL | fee_decay_price_weakness_context | -1 | 10.7636 |  |  |  |  | pending_4h |
| 2026-06-08T00:46:00+00:00 | SOL | fee_decay_price_weakness_context | -1 | 9.6990 |  |  |  |  | pending_4h |
| 2026-06-08T09:52:00+00:00 | SOL | fee_decay_price_weakness_context | -1 | 9.6990 |  |  |  |  | pending_4h |
