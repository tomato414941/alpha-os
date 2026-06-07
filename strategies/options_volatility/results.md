# Options Volatility Results

Run:

```bash
uv run python -m strategies.options_volatility.current_deribit_options_surface
```

This lane compresses public Deribit BTC/ETH option summaries into ATM IV,
simple 5% OTM skew, and adjacent-expiry term structure. It is a volatility
surface exploration probe, not a trade instruction.

## Current Deribit Options Surface

| currency | expiry | dte | atm iv | skew iv | term iv spread | spread pct | oi | volume USD | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| BTC | 2026-06-08 | 0.63 | 64.68 | 20.42 | -9.26 | 0.2593 | 4889 | 889676 | put_skew_watch | 48.0600 |
| BTC | 2026-06-12 | 4.63 | 64.15 | 15.46 | 10.39 | 0.0519 | 30582 | 1200670 | put_skew_watch | 46.7012 |
| ETH | 2026-06-12 | 4.63 | 80.84 | 10.33 | 10.91 | 0.0447 | 136986 | 192091 | front_vol_premium_watch | 42.4809 |
| ETH | 2026-06-08 | 0.63 | 84.72 | 23.44 | -2.71 | 0.1316 | 52768 | 289041 | put_skew_watch | 38.7802 |
| BTC | 2026-06-09 | 1.63 | 73.94 | 20.32 | 4.42 | 0.1028 | 3174 | 935784 | put_skew_watch | 38.4273 |
| BTC | 2026-06-10 | 2.63 | 69.52 | 19.06 | 2.96 | 0.0548 | 961 | 487311 | put_skew_watch | 33.5411 |
| ETH | 2026-06-09 | 1.63 | 87.43 | 14.77 | 3.72 | 0.0783 | 33526 | 172778 | put_skew_watch | 31.8163 |
| BTC | 2026-06-26 | 18.63 | 49.70 | 6.69 | 4.50 | 0.0283 | 143587 | 2238139 | put_skew_watch | 27.1405 |
| ETH | 2026-06-26 | 18.63 | 66.08 | 4.72 | 5.60 | 0.0331 | 904860 | 207576 | front_vol_premium_watch | 27.1275 |
| BTC | 2026-06-19 | 11.63 | 53.76 | 8.96 | 4.06 | 0.0321 | 15386 | 503135 | put_skew_watch | 26.9046 |

Interpretation:

- Short-dated BTC and ETH puts are materially richer than the simple 5% OTM
  call proxy in this snapshot.
- ETH 2026-06-12 and 2026-06-26 show front-vol premium against the next expiry.
- This is not yet alpha evidence. The next step is realized-vol labeling,
  option execution-cost checks, margin, and hedge-cost modeling.
