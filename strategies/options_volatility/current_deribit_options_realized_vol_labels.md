# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH | 2026-06-09 | 0.70 | cheap_vol_watch | 61.72 | 85.56 | 112.03 | -23.84 | -50.31 | 14.11 | -3.09 | 67.5072 |
| ETH | 2026-06-10 | 1.70 | cheap_vol_watch | 64.81 | 85.56 | 112.03 | -20.75 | -47.22 | 13.86 | -1.52 | 62.5972 |
| ETH | 2026-06-26 | 17.70 | cheap_vol_watch | 59.47 | 85.56 | 112.03 | -26.09 | -52.56 | 5.98 | 2.72 | 61.2572 |
| ETH | 2026-06-12 | 3.70 | cheap_vol_watch | 65.68 | 85.56 | 112.03 | -19.88 | -46.35 | 9.43 | 4.24 | 60.0172 |
| ETH | 2026-06-19 | 10.70 | cheap_vol_watch | 61.44 | 85.56 | 112.03 | -24.12 | -50.59 | 7.30 | 1.97 | 59.8572 |
| ETH | 2026-07-31 | 52.70 | cheap_vol_watch | 56.75 | 85.56 | 112.03 | -28.81 | -55.28 | 3.31 | -0.11 | 58.6972 |
| ETH | 2026-09-25 | 108.70 | cheap_vol_watch | 57.08 | 85.56 | 112.03 | -28.48 | -54.95 | 1.63 | -1.98 | 58.5572 |
| ETH | 2026-08-28 | 80.70 | cheap_vol_watch | 56.86 | 85.56 | 112.03 | -28.70 | -55.17 | 2.30 | -0.22 | 57.6872 |
| BTC | 2026-06-09 | 0.70 | cheap_vol_watch | 48.63 | 88.22 | 86.12 | -39.59 | -37.49 | 19.58 | 0.53 | 57.5960 |
| BTC | 2026-06-10 | 1.70 | cheap_vol_watch | 48.10 | 88.22 | 86.12 | -40.12 | -38.02 | 18.05 | -1.46 | 57.5260 |
| ETH | 2026-06-11 | 2.70 | cheap_vol_watch | 66.33 | 85.56 | 112.03 | -19.23 | -45.70 | 11.05 | 0.65 | 57.3972 |
| ETH | 2026-12-25 | 199.70 | cheap_vol_watch | 59.06 | 85.56 | 112.03 | -26.50 | -52.97 | 1.28 | -1.40 | 55.6472 |
| BTC | 2026-06-12 | 3.70 | cheap_vol_watch | 48.19 | 88.22 | 86.12 | -40.03 | -37.93 | 13.80 | 3.59 | 55.3160 |
| BTC | 2026-06-11 | 2.70 | cheap_vol_watch | 49.56 | 88.22 | 86.12 | -38.66 | -36.56 | 14.50 | 1.37 | 52.4260 |
| ETH | 2027-03-26 | 290.70 | cheap_vol_watch | 60.46 | 85.56 | 112.03 | -25.10 | -51.57 | 0.48 |  | 52.0472 |
| BTC | 2026-06-19 | 10.70 | cheap_vol_watch | 44.60 | 88.22 | 86.12 | -43.62 | -41.52 | 7.65 | 2.27 | 51.4360 |
| BTC | 2026-06-26 | 17.70 | cheap_vol_watch | 42.33 | 88.22 | 86.12 | -45.89 | -43.79 | 6.06 | 1.01 | 50.8560 |
| BTC | 2026-09-25 | 108.70 | cheap_vol_watch | 41.52 | 88.22 | 86.12 | -46.70 | -44.60 | 1.77 | -2.16 | 48.5260 |
| BTC | 2026-07-31 | 52.70 | cheap_vol_watch | 41.32 | 88.22 | 86.12 | -46.90 | -44.80 | 3.46 | -0.22 | 48.4760 |
| BTC | 2026-08-28 | 80.70 | cheap_vol_watch | 41.54 | 88.22 | 86.12 | -46.68 | -44.58 | 2.79 | 0.02 | 47.3860 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
