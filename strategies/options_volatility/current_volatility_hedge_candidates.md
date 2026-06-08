# Current Volatility Hedge Candidates

This turns option actionability rows into hedge-plan candidates. It is not a live options execution instruction.

| candidate | decision | score | max loss % | max loss USD | prem/rv move | spread | depth USD | hedge interval | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| btc_20260619_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 71.4994 | 6.85 | 4309.31 | 0.47 | 0.0296 | 124970 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| eth_20260619_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 69.3319 | 9.35 | 156.36 | 0.49 | 0.0326 | 60666 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| eth_20260626_medium_dated_delta_hedge_check | paper_delta_hedge_candidate | 69.0839 | 11.35 | 190.01 | 0.47 | 0.0313 | 96336 | daily_or_large_delta_move | quote and premium are good enough to test medium_dated_delta_hedge_check with explicit hedge PnL |
| btc_20260612_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 66.7914 | 4.80 | 3019.29 | 0.55 | 0.0426 | 51026 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| btc_20260626_medium_dated_delta_hedge_check | quote_only_hedge_watch | 50.3102 | 8.30 | 5224.51 | 0.44 | 0.0306 | 1567 | daily_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| btc_20260611_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 50.2827 | 4.30 | 2705.15 | 0.58 | 0.0476 | 53021 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| eth_20260610_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 48.5629 | 4.40 | 73.65 | 0.57 | 0.0706 | 29460 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| eth_20260612_short_dated_delta_hedge_check | quote_only_hedge_watch | 48.0233 | 6.20 | 103.74 | 0.55 | 0.0667 | 41494 | 4h_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| btc_20260610_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 45.5895 | 3.65 | 2296.23 | 0.61 | 0.0709 | 5970 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |

## Interpretation

Long-vol alpha is only useful if the option quote, premium-at-risk, delta hedge path, exit bid, and margin treatment are explicit. Rows here identify which structures are ready for that paper hedge check.
