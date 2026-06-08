# Current Volatility Hedge Candidates

This turns option actionability rows into hedge-plan candidates. It is not a live options execution instruction.

| candidate | decision | score | max loss % | max loss USD | prem/rv move | spread | depth USD | hedge interval | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| eth_20260619_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 69.4783 | 9.30 | 155.37 | 0.49 | 0.0272 | 62148 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| eth_20260626_medium_dated_delta_hedge_check | paper_delta_hedge_candidate | 68.4226 | 11.30 | 188.92 | 0.47 | 0.0269 | 73678 | daily_or_large_delta_move | quote and premium are good enough to test medium_dated_delta_hedge_check with explicit hedge PnL |
| btc_20260612_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 67.7404 | 4.70 | 2970.04 | 0.56 | 0.0435 | 87319 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| btc_20260611_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 50.7025 | 4.20 | 2654.15 | 0.58 | 0.0488 | 78297 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| btc_20260626_medium_dated_delta_hedge_check | quote_only_hedge_watch | 50.0486 | 8.20 | 5185.59 | 0.45 | 0.0247 | 1556 | daily_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| btc_20260619_short_dated_delta_hedge_check | quote_only_hedge_watch | 49.7393 | 6.80 | 4297.99 | 0.48 | 0.0299 | 1289 | 4h_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| btc_20260610_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 49.3453 | 3.45 | 2180.18 | 0.59 | 0.0444 | 654 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| eth_20260612_short_dated_delta_hedge_check | quote_only_hedge_watch | 49.0914 | 6.05 | 101.07 | 0.54 | 0.0508 | 6064 | 4h_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |

## Interpretation

Long-vol alpha is only useful if the option quote, premium-at-risk, delta hedge path, exit bid, and margin treatment are explicit. Rows here identify which structures are ready for that paper hedge check.
