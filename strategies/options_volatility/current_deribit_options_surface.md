# Current Deribit Options Surface

This compresses public Deribit BTC/ETH option summaries into ATM IV, simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility-surface exploration probe, not a trade instruction.

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BTC | 2026-06-09 | 0.50 | 38.88 | 17.14 | -6.68 | 0.2976 | 6655 | 3113438 | put_skew_watch | 40.2212 |
| ETH | 2026-06-09 | 0.50 | 58.37 | 13.33 | -4.16 | 0.2583 | 50944 | 484511 | put_skew_watch | 31.5257 |
| BTC | 2026-06-10 | 1.50 | 45.56 | 18.18 | -2.01 | 0.2500 | 4217 | 1208650 | put_skew_watch | 31.4074 |
| ETH | 2026-06-12 | 3.50 | 64.28 | 11.42 | 4.22 | 0.0727 | 147019 | 875467 | put_skew_watch | 30.8243 |
| BTC | 2026-06-12 | 3.50 | 47.97 | 15.86 | 1.86 | 0.0921 | 31683 | 2872230 | put_skew_watch | 30.3548 |
| ETH | 2026-06-10 | 1.50 | 62.53 | 15.46 | -2.19 | 0.1346 | 16917 | 363776 | put_skew_watch | 29.3600 |
| BTC | 2026-06-11 | 2.50 | 47.57 | 15.90 | -0.40 | 0.1003 | 3014 | 1120226 | put_skew_watch | 26.0280 |
| BTC | 2026-06-19 | 10.50 | 46.11 | 8.65 | 2.09 | 0.0519 | 17313 | 3986796 | put_skew_watch | 23.5652 |
| ETH | 2026-06-11 | 2.50 | 64.72 | 12.97 | 0.44 | 0.0749 | 8538 | 259884 | put_skew_watch | 23.0464 |
| BTC | 2026-06-26 | 17.50 | 44.02 | 6.75 | 2.02 | 0.0375 | 145195 | 5995160 | put_skew_watch | 22.6548 |
| ETH | 2026-06-26 | 17.50 | 58.31 | 5.09 | 2.13 | 0.0300 | 914675 | 1448538 | put_skew_watch | 21.4121 |
| ETH | 2026-06-19 | 10.50 | 60.06 | 6.15 | 1.75 | 0.0339 | 61477 | 677920 | put_skew_watch | 20.2022 |
| BTC | 2026-09-25 | 108.50 | 42.27 | 2.48 | -1.81 | 0.0179 | 76665 | 4134340 | surface_context | 17.5652 |
| ETH | 2026-09-25 | 108.50 | 56.68 | 1.56 | -2.12 | 0.0139 | 295194 | 920825 | surface_context | 17.2065 |
| ETH | 2026-12-25 | 199.50 | 58.80 | 1.15 | -1.39 | 0.0141 | 396890 | 573690 | surface_context | 15.2592 |
| BTC | 2026-07-31 | 52.50 | 42.00 | 3.78 | -0.02 | 0.0140 | 38984 | 5355355 | surface_context | 15.1117 |
| BTC | 2026-12-25 | 199.50 | 44.08 | 1.13 | -0.85 | 0.0136 | 80493 | 10617591 | surface_context | 14.7345 |
| ETH | 2026-07-31 | 52.50 | 56.18 | 2.48 | -0.24 | 0.0183 | 144968 | 1344911 | surface_context | 14.2133 |
| BTC | 2026-08-28 | 80.50 | 42.02 | 2.39 | -0.25 | 0.0188 | 7042 | 1556495 | surface_context | 12.8924 |
| ETH | 2026-08-28 | 80.50 | 56.42 | 2.05 | -0.26 | 0.0134 | 18189 | 350787 | surface_context | 12.3480 |

## Interpretation

Large positive term spread means the nearer expiry has richer ATM IV than the next expiry. Positive skew means the 5% OTM put proxy is richer than the 5% OTM call proxy. This still needs realized-vol baselines, option execution costs, margin, and hedging rules.
