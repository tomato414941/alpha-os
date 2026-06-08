# Protocol Fee Price-Lag Forward Labels

This labels stored fee-growth price-lag observations. Positive directional return means the observation's direction was right before fees, funding PnL, and slippage.

- total rows: `34`
- labeled 4h rows: `6`

| observed at | token | status | dir | priority | dir 4h | dir 12h | dir 24h | dir 7d | label status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-07T22:36:00+00:00 | CRV | fee_growth_price_lag_candidate | 1 | 59.1753 | 0.00396679 | 0.01829538 |  |  | labeled_12h_pending_24h |
| 2026-06-08T00:41:00+00:00 | CRV | fee_growth_price_lag_candidate | 1 | 59.1753 | -0.00023355 |  |  |  | labeled_4h_pending_12h |
| 2026-06-08T00:46:00+00:00 | CRV | fee_growth_price_lag_candidate | 1 | 59.9474 | -0.00697578 |  |  |  | labeled_4h_pending_12h |
| 2026-06-07T22:36:00+00:00 | AAVE | fee_growth_price_lag_candidate | 1 | 87.8950 | -0.01479578 | -0.01203360 |  |  | labeled_12h_pending_24h |
| 2026-06-08T00:41:00+00:00 | AAVE | fee_growth_price_lag_candidate | 1 | 87.8950 | -0.02321015 |  |  |  | labeled_4h_pending_12h |
| 2026-06-08T00:46:00+00:00 | AAVE | fee_growth_price_lag_candidate | 1 | 87.9800 | -0.02549486 |  |  |  | labeled_4h_pending_12h |
| 2026-06-07T22:36:00+00:00 | JUP | fee_growth_price_lag_candidate | 1 | 94.3453 |  |  |  |  | pending_4h |
| 2026-06-08T00:41:00+00:00 | JUP | fee_growth_price_lag_candidate | 1 | 94.3453 |  |  |  |  | pending_4h |
| 2026-06-08T00:46:00+00:00 | JUP | fee_growth_price_lag_candidate | 1 | 93.9263 |  |  |  |  | pending_4h |
| 2026-06-08T09:52:00+00:00 | JUP | fee_growth_price_lag_candidate | 1 | 93.9263 |  |  |  |  | pending_4h |
| 2026-06-08T09:52:00+00:00 | AAVE | fee_growth_price_lag_candidate | 1 | 87.9800 |  |  |  |  | pending_4h |
| 2026-06-07T22:36:00+00:00 | UNI | fee_growth_price_lag_candidate | 1 | 77.9108 |  |  |  |  | pending_4h |
| 2026-06-08T00:41:00+00:00 | UNI | fee_growth_price_lag_candidate | 1 | 77.9108 |  |  |  |  | pending_4h |
| 2026-06-08T00:46:00+00:00 | UNI | fee_growth_price_lag_candidate | 1 | 77.8233 |  |  |  |  | pending_4h |
| 2026-06-08T09:52:00+00:00 | UNI | fee_growth_price_lag_candidate | 1 | 77.8233 |  |  |  |  | pending_4h |
| 2026-06-07T22:36:00+00:00 | MORPHO | fee_growth_price_lag_candidate | 1 | 60.8602 |  |  |  |  | pending_4h |
| 2026-06-08T00:41:00+00:00 | MORPHO | fee_growth_price_lag_candidate | 1 | 60.8602 |  |  |  |  | pending_4h |
| 2026-06-08T09:52:00+00:00 | CRV | fee_growth_price_lag_candidate | 1 | 59.9474 |  |  |  |  | pending_4h |
| 2026-06-08T00:46:00+00:00 | MORPHO | fee_growth_price_lag_candidate | 1 | 58.5616 |  |  |  |  | pending_4h |
| 2026-06-08T09:52:00+00:00 | MORPHO | fee_growth_price_lag_candidate | 1 | 58.5616 |  |  |  |  | pending_4h |
| 2026-06-08T00:46:00+00:00 | PENDLE | fee_growth_price_lag_candidate | 1 | 40.2142 |  |  |  |  | pending_4h |
| 2026-06-08T09:52:00+00:00 | PENDLE | fee_growth_price_lag_candidate | 1 | 40.2142 |  |  |  |  | pending_4h |
| 2026-06-07T22:36:00+00:00 | PENDLE | fee_growth_price_lag_candidate | 1 | 39.2389 |  |  |  |  | pending_4h |
| 2026-06-08T00:41:00+00:00 | PENDLE | fee_growth_price_lag_candidate | 1 | 39.2389 |  |  |  |  | pending_4h |
| 2026-06-07T22:36:00+00:00 | UNI | fee_decay_price_weakness_context | -1 | 32.1903 |  |  |  |  | pending_4h |
| 2026-06-08T00:41:00+00:00 | UNI | fee_decay_price_weakness_context | -1 | 32.1903 |  |  |  |  | pending_4h |
| 2026-06-08T00:46:00+00:00 | UNI | fee_decay_price_weakness_context | -1 | 31.9772 |  |  |  |  | pending_4h |
| 2026-06-08T09:52:00+00:00 | UNI | fee_decay_price_weakness_context | -1 | 31.9772 |  |  |  |  | pending_4h |
| 2026-06-08T00:46:00+00:00 | HYPE | fee_decay_price_weakness_context | -1 | 11.2882 |  |  |  |  | pending_4h |
| 2026-06-08T09:52:00+00:00 | HYPE | fee_decay_price_weakness_context | -1 | 11.2882 |  |  |  |  | pending_4h |
| 2026-06-07T22:36:00+00:00 | SOL | fee_decay_price_weakness_context | -1 | 10.7636 |  |  |  |  | pending_4h |
| 2026-06-08T00:41:00+00:00 | SOL | fee_decay_price_weakness_context | -1 | 10.7636 |  |  |  |  | pending_4h |
| 2026-06-08T00:46:00+00:00 | SOL | fee_decay_price_weakness_context | -1 | 9.6990 |  |  |  |  | pending_4h |
| 2026-06-08T09:52:00+00:00 | SOL | fee_decay_price_weakness_context | -1 | 9.6990 |  |  |  |  | pending_4h |
