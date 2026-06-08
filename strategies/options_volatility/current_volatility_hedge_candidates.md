# Current Volatility Hedge Candidates

This turns option actionability rows into hedge-plan candidates. It is not a live options execution instruction.

| candidate | decision | score | max loss % | max loss USD | prem/rv move | spread | depth USD | hedge interval | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| btc_20260626_medium_dated_delta_hedge_check | paper_delta_hedge_candidate | 71.9159 | 7.55 | 4838.94 | 0.40 | 0.0405 | 149039 | daily_or_large_delta_move | quote and premium are good enough to test medium_dated_delta_hedge_check with explicit hedge PnL |
| btc_20260619_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 71.2484 | 6.20 | 3972.49 | 0.42 | 0.0412 | 122750 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| btc_20260611_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 51.0196 | 3.50 | 2242.95 | 0.47 | 0.0588 | 63924 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| eth_20260619_short_dated_delta_hedge_check | quote_only_hedge_watch | 50.9263 | 8.55 | 145.32 | 0.45 | 0.0297 | 9736 | 4h_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| btc_20260610_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 50.7526 | 2.70 | 1730.28 | 0.46 | 0.0571 | 40488 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| eth_20260626_medium_dated_delta_hedge_check | quote_only_hedge_watch | 50.4946 | 10.55 | 179.44 | 0.43 | 0.0191 | 10049 | daily_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| eth_20260611_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 49.1021 | 4.70 | 79.93 | 0.49 | 0.0659 | 10710 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| btc_20260612_short_dated_delta_hedge_check | quote_only_hedge_watch | 47.9729 | 4.00 | 2562.38 | 0.46 | 0.0779 | 86608 | 4h_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| eth_20260610_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 46.8932 | 3.70 | 62.91 | 0.48 | 0.0845 | 11702 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |

## Interpretation

Long-vol alpha is only useful if the option quote, premium-at-risk, delta hedge path, exit bid, and margin treatment are explicit. Rows here identify which structures are ready for that paper hedge check.
