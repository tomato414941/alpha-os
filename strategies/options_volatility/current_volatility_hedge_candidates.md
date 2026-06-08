# Current Volatility Hedge Candidates

This turns option actionability rows into hedge-plan candidates. It is not a live options execution instruction.

| candidate | decision | score | max loss % | max loss USD | prem/rv move | spread | depth USD | hedge interval | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| eth_20260619_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 70.5620 | 9.35 | 156.11 | 0.49 | 0.0326 | 97570 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| eth_20260626_medium_dated_delta_hedge_check | paper_delta_hedge_candidate | 69.2098 | 11.35 | 189.63 | 0.47 | 0.0268 | 98795 | daily_or_large_delta_move | quote and premium are good enough to test medium_dated_delta_hedge_check with explicit hedge PnL |
| btc_20260612_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 67.7781 | 4.70 | 2964.85 | 0.55 | 0.0435 | 87167 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| btc_20260611_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 51.0429 | 4.20 | 2649.50 | 0.58 | 0.0488 | 94057 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| eth_20260612_short_dated_delta_hedge_check | quote_only_hedge_watch | 50.6132 | 6.05 | 101.01 | 0.54 | 0.0336 | 16667 | 4h_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| btc_20260626_medium_dated_delta_hedge_check | quote_only_hedge_watch | 50.0176 | 8.20 | 5176.95 | 0.45 | 0.0247 | 518 | daily_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| btc_20260619_short_dated_delta_hedge_check | quote_only_hedge_watch | 49.7867 | 6.85 | 4322.02 | 0.48 | 0.0296 | 3025 | 4h_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |

## Interpretation

Long-vol alpha is only useful if the option quote, premium-at-risk, delta hedge path, exit bid, and margin treatment are explicit. Rows here identify which structures are ready for that paper hedge check.
