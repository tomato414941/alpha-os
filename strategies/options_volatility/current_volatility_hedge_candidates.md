# Current Volatility Hedge Candidates

This turns option actionability rows into hedge-plan candidates. It is not a live options execution instruction.

| candidate | decision | score | max loss % | max loss USD | prem/rv move | spread | depth USD | hedge interval | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| btc_20260626_medium_dated_delta_hedge_check | paper_delta_hedge_candidate | 72.8637 | 7.75 | 4915.65 | 0.41 | 0.0195 | 193185 | daily_or_large_delta_move | quote and premium are good enough to test medium_dated_delta_hedge_check with explicit hedge PnL |
| btc_20260619_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 72.2611 | 6.35 | 4026.02 | 0.44 | 0.0320 | 163054 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| eth_20260612_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 67.3645 | 5.15 | 86.72 | 0.47 | 0.0498 | 45790 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| btc_20260610_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 52.6885 | 2.40 | 1521.17 | 0.44 | 0.0426 | 49742 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| btc_20260611_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 51.8352 | 3.25 | 2059.92 | 0.46 | 0.0472 | 35225 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| eth_20260619_short_dated_delta_hedge_check | quote_only_hedge_watch | 51.1139 | 8.30 | 139.76 | 0.44 | 0.0306 | 16772 | 4h_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| eth_20260626_medium_dated_delta_hedge_check | quote_only_hedge_watch | 50.9915 | 10.40 | 175.24 | 0.42 | 0.0243 | 24183 | daily_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| btc_20260612_short_dated_delta_hedge_check | quote_only_hedge_watch | 50.2784 | 3.85 | 2440.16 | 0.46 | 0.0397 | 2928 | 4h_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| eth_20260611_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 49.8714 | 4.40 | 74.14 | 0.47 | 0.0585 | 4448 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| eth_20260610_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 48.0531 | 3.35 | 56.44 | 0.46 | 0.0775 | 27771 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |

## Interpretation

Long-vol alpha is only useful if the option quote, premium-at-risk, delta hedge path, exit bid, and margin treatment are explicit. Rows here identify which structures are ready for that paper hedge check.
