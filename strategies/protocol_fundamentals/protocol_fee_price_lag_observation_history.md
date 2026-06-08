# Protocol Fee Price-Lag Observation History

This stores current protocol fee-growth price-lag observations so later runs can attach 4h, 12h, 24h, and 7d forward labels. It is a sample store, not a trade log.

- total rows: `25`
- ready rows: `25`

| observed at | token | status | dir | price | fee growth 7d | price 7d | score | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-08T00:46:00+00:00 | AAVE | fee_growth_price_lag_candidate | 1 | 64.13000000 | 129.80 | -21.82 | 87.9800 | paper-label AAVE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-08T00:46:00+00:00 | CRV | fee_growth_price_lag_candidate | 1 | 0.19766300 | 210.10 | -8.57 | 59.9474 | paper-label CRV as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-08T00:46:00+00:00 | HYPE | fee_decay_price_weakness_context | -1 | 61.68000000 | -10.50 | -14.98 | 11.2882 | test whether HYPE fee decay and weak price persist before any short thesis |
| 2026-06-08T00:46:00+00:00 | JUP | fee_growth_price_lag_candidate | 1 | 0.15897500 | 204.43 | -16.78 | 93.9263 | paper-label JUP as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-08T00:46:00+00:00 | MORPHO | fee_growth_price_lag_candidate | 1 | 1.75000000 | 140.56 | -19.06 | 58.5616 | paper-label MORPHO as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-08T00:46:00+00:00 | PENDLE | fee_growth_price_lag_candidate | 1 | 1.28000000 | 158.55 | -5.79 | 40.2142 | paper-label PENDLE as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-08T00:46:00+00:00 | SOL | fee_decay_price_weakness_context | -1 | 66.81000000 | -18.52 | -18.76 | 9.6990 | test whether SOL fee decay and weak price persist before any short thesis |
| 2026-06-08T00:46:00+00:00 | UNI | fee_growth_price_lag_candidate | 1 | 2.58000000 | 105.16 | -14.58 | 77.8233 | paper-label UNI as fee-growth lag setup over 4h, 12h, 24h, and 7d |
| 2026-06-08T00:46:00+00:00 | UNI | fee_decay_price_weakness_context | -1 | 2.58000000 | -33.04 | -14.58 | 31.9772 | test whether UNI fee decay and weak price persist before any short thesis |
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
