# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BTC | 2026-06-08 | 0.41 | 67.77 | 31.73 | -12.30 | 0.3110 | 5071 | 964480 | put_skew_watch | 65.3976 |
| BTC | 2026-06-12 | 4.41 | 67.70 | 14.77 | 12.30 | 0.0593 | 30596 | 1158560 | put_skew_watch | 49.8010 |
| ETH | 2026-06-08 | 0.41 | 87.79 | 30.62 | -2.28 | 0.1907 | 52875 | 288229 | put_skew_watch | 44.9815 |
| ETH | 2026-06-12 | 4.41 | 81.70 | 10.94 | 11.15 | 0.0699 | 138287 | 330201 | front_vol_premium_watch | 43.7598 |
| BTC | 2026-06-09 | 1.41 | 80.07 | 23.81 | 4.36 | 0.0778 | 3303 | 919756 | put_skew_watch | 41.8572 |
| BTC | 2026-06-10 | 2.41 | 75.71 | 21.59 | 4.54 | 0.0670 | 1159 | 607132 | put_skew_watch | 39.3835 |
| ETH | 2026-06-09 | 1.41 | 90.07 | 17.36 | 3.49 | 0.0974 | 33836 | 142077 | put_skew_watch | 33.8270 |
| BTC | 2026-06-11 | 3.41 | 71.17 | 14.78 | 3.47 | 0.0591 | 162 | 164354 | put_skew_watch | 29.0292 |
| ETH | 2026-06-26 | 18.41 | 66.73 | 5.04 | 6.15 | 0.0329 | 904921 | 142612 | front_vol_premium_watch | 28.3850 |
| BTC | 2026-06-19 | 11.41 | 55.40 | 8.60 | 4.73 | 0.0310 | 15818 | 838443 | put_skew_watch | 28.1207 |
| BTC | 2026-06-26 | 18.41 | 50.67 | 6.50 | 4.89 | 0.0309 | 143610 | 2802800 | put_skew_watch | 27.8231 |
| ETH | 2026-06-10 | 2.41 | 86.58 | 13.26 | 2.87 | 0.0592 | 5022 | 88143 | put_skew_watch | 27.5278 |
| ETH | 2026-06-19 | 11.41 | 70.55 | 6.42 | 3.82 | 0.0325 | 54426 | 83872 | put_skew_watch | 23.6545 |
| ETH | 2026-06-11 | 3.41 | 83.71 | 11.12 | 2.01 | 0.0627 | 1095 | 43613 | put_skew_watch | 22.6940 |
| BTC | 2026-07-31 | 53.41 | 45.78 | 3.46 | 0.95 | 0.0175 | 37203 | 1367235 | surface_context | 16.0314 |
| ETH | 2026-07-31 | 53.41 | 60.58 | 2.43 | 1.02 | 0.0196 | 139641 | 134711 | surface_context | 14.7053 |
| BTC | 2026-12-25 | 200.41 | 45.07 | 1.29 | -0.69 | 0.0193 | 80428 | 1619853 | surface_context | 13.7464 |
| BTC | 2026-09-25 | 109.41 | 44.69 | 1.81 | -0.38 | 0.0187 | 76377 | 1080374 | surface_context | 13.4492 |
| ETH | 2026-12-25 | 200.41 | 60.19 | 0.55 | -1.07 | 0.0208 | 394504 | 58845 | surface_context | 13.0143 |
| ETH | 2026-09-25 | 109.41 | 59.81 | 0.80 | -0.38 | 0.0196 | 293162 | 187108 | surface_context | 12.2600 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
