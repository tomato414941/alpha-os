# Protocol Fee Price-Lag Observation History

This stores current protocol fee-growth price-lag observations so later runs can attach 4h, 12h, 24h, and 7d forward labels. It is a sample store, not a trade log.

- total rows: `16`
- ready rows: `16`

| observed at | token | status | dir | price | fee growth 7d | price 7d | score | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-08T00:41:00+00:00 | AAVE | fee_growth_price_lag_candidate | 1 | 63.98000000 | 128.95 | -21.58 | 87.8950 | paper-label AAVE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-08T00:41:00+00:00 | CRV | fee_growth_price_lag_candidate | 1 | 0.19633000 | 190.42 | -8.99 | 59.1753 | paper-label CRV as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-08T00:41:00+00:00 | JUP | fee_growth_price_lag_candidate | 1 | 0.15711800 | 196.60 | -16.66 | 94.3453 | paper-label JUP as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-08T00:41:00+00:00 | MORPHO | fee_growth_price_lag_candidate | 1 | 1.72000000 | 151.57 | -19.85 | 60.8602 | paper-label MORPHO as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-08T00:41:00+00:00 | PENDLE | fee_growth_price_lag_candidate | 1 | 1.26000000 | 118.68 | -8.03 | 39.2389 | paper-label PENDLE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-08T00:41:00+00:00 | SOL | fee_decay_price_weakness_context | -1 | 66.16000000 | -8.44 | -19.08 | 10.7636 | test whether SOL fee decay and weak price persist before any short thesis |
| 2026-06-08T00:41:00+00:00 | UNI | fee_growth_price_lag_candidate | 1 | 2.56000000 | 95.01 | -14.93 | 77.9108 | paper-label UNI as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-08T00:41:00+00:00 | UNI | fee_decay_price_weakness_context | -1 | 2.56000000 | -38.41 | -14.93 | 32.1903 | test whether UNI fee decay and weak price persist before any short thesis |
| 2026-06-07T22:36:00+00:00 | AAVE | fee_growth_price_lag_candidate | 1 | 63.98000000 | 128.95 | -21.58 | 87.8950 | paper-label AAVE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-07T22:36:00+00:00 | CRV | fee_growth_price_lag_candidate | 1 | 0.19633000 | 190.42 | -8.99 | 59.1753 | paper-label CRV as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-07T22:36:00+00:00 | JUP | fee_growth_price_lag_candidate | 1 | 0.15711800 | 196.60 | -16.66 | 94.3453 | paper-label JUP as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-07T22:36:00+00:00 | MORPHO | fee_growth_price_lag_candidate | 1 | 1.72000000 | 151.57 | -19.85 | 60.8602 | paper-label MORPHO as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-07T22:36:00+00:00 | PENDLE | fee_growth_price_lag_candidate | 1 | 1.26000000 | 118.68 | -8.03 | 39.2389 | paper-label PENDLE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-07T22:36:00+00:00 | SOL | fee_decay_price_weakness_context | -1 | 66.16000000 | -8.44 | -19.08 | 10.7636 | test whether SOL fee decay and weak price persist before any short thesis |
| 2026-06-07T22:36:00+00:00 | UNI | fee_growth_price_lag_candidate | 1 | 2.56000000 | 95.01 | -14.93 | 77.9108 | paper-label UNI as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-07T22:36:00+00:00 | UNI | fee_decay_price_weakness_context | -1 | 2.56000000 | -38.41 | -14.93 | 32.1903 | test whether UNI fee decay and weak price persist before any short thesis |
