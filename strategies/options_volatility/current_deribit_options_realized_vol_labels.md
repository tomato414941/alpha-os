# Current Deribit Options Realized Vol Labels

This joins Deribit ATM IV to recent Hyperliquid 15m realized volatility. It is a fast IV-vs-realized context label, not an options backtest.

| currency | expiry | dte | action | atm iv | rv 4h | rv 24h | prem 4h | prem 24h | skew | term | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 2026-06-09 | 0.30 | cheap_vol_watch | 44.37 | 57.21 | 57.42 | -12.84 | -13.05 | 17.66 | -6.32 | 37.0294 |
| ETH | 2026-06-09 | 0.30 | realized_vol_context | 66.95 | 98.83 | 72.32 | -31.88 | -5.37 | 22.62 | 0.24 | 28.2293 |
| ETH | 2026-06-10 | 1.30 | realized_vol_context | 66.71 | 98.83 | 72.32 | -32.12 | -5.61 | 17.85 | 0.44 | 23.8993 |
| BTC | 2026-06-10 | 1.30 | realized_vol_context | 50.69 | 57.21 | 57.42 | -6.52 | -6.73 | 16.18 | -0.16 | 23.0694 |
| BTC | 2026-06-12 | 3.30 | realized_vol_context | 50.51 | 57.21 | 57.42 | -6.70 | -6.91 | 11.15 | 4.29 | 22.3494 |
| ETH | 2026-06-12 | 3.30 | realized_vol_context | 65.65 | 98.83 | 72.32 | -33.18 | -6.67 | 12.31 | 3.25 | 22.2293 |
| ETH | 2026-06-11 | 2.30 | realized_vol_context | 66.27 | 98.83 | 72.32 | -32.56 | -6.05 | 14.93 | 0.62 | 21.5993 |
| BTC | 2026-06-26 | 17.30 | cheap_vol_watch | 44.31 | 57.21 | 57.42 | -12.90 | -13.11 | 5.62 | 1.10 | 19.8294 |
| BTC | 2026-06-19 | 10.30 | cheap_vol_watch | 46.22 | 57.21 | 57.42 | -10.99 | -11.20 | 6.69 | 1.91 | 19.7994 |
| ETH | 2026-06-26 | 17.30 | cheap_vol_watch | 60.18 | 98.83 | 72.32 | -38.65 | -12.14 | 4.59 | 3.01 | 19.7393 |
| ETH | 2026-09-25 | 108.30 | cheap_vol_watch | 56.76 | 98.83 | 72.32 | -42.07 | -15.56 | 1.51 | -2.17 | 19.2393 |
| BTC | 2026-06-11 | 2.30 | realized_vol_context | 50.85 | 57.21 | 57.42 | -6.36 | -6.57 | 12.04 | 0.34 | 18.9494 |
| BTC | 2026-09-25 | 108.30 | cheap_vol_watch | 42.59 | 57.21 | 57.42 | -14.62 | -14.83 | 1.80 | -1.86 | 18.4894 |
| ETH | 2026-07-31 | 52.30 | cheap_vol_watch | 57.17 | 98.83 | 72.32 | -41.66 | -15.15 | 2.30 | 0.64 | 18.0893 |
| ETH | 2026-08-28 | 80.30 | cheap_vol_watch | 56.53 | 98.83 | 72.32 | -42.30 | -15.79 | 1.54 | -0.23 | 17.5593 |
| BTC | 2026-07-31 | 52.30 | cheap_vol_watch | 43.21 | 57.21 | 57.42 | -14.00 | -14.21 | 3.01 | 0.27 | 17.4894 |
| ETH | 2026-06-19 | 10.30 | realized_vol_context | 62.40 | 98.83 | 72.32 | -36.43 | -9.92 | 5.20 | 2.22 | 17.3393 |
| BTC | 2026-08-28 | 80.30 | cheap_vol_watch | 42.94 | 57.21 | 57.42 | -14.27 | -14.48 | 2.11 | 0.35 | 16.9394 |
| ETH | 2026-12-25 | 199.30 | cheap_vol_watch | 58.93 | 98.83 | 72.32 | -39.90 | -13.39 | 1.06 | -1.31 | 15.7593 |
| BTC | 2026-12-25 | 199.30 | cheap_vol_watch | 44.45 | 57.21 | 57.42 | -12.76 | -12.97 | 1.60 | -0.58 | 15.1494 |

## Interpretation

Positive IV premium means listed ATM IV is above recent realized volatility. This can point to vol-selling or event-premium candidates, but it still needs realized-vol forecasts, hedge PnL, option spreads, margin, and tail-risk controls.
