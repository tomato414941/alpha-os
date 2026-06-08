# Current Volatility Hedge Candidates

This turns option actionability rows into hedge-plan candidates. It is not a live options execution instruction.

| candidate | decision | score | max loss % | max loss USD | prem/rv move | spread | depth USD | hedge interval | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| btc_20260619_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 71.1306 | 6.80 | 4313.04 | 0.46 | 0.0299 | 106532 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| eth_20260619_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 70.4216 | 9.05 | 152.39 | 0.47 | 0.0337 | 92655 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| eth_20260626_medium_dated_delta_hedge_check | paper_delta_hedge_candidate | 69.1841 | 11.05 | 186.24 | 0.45 | 0.0229 | 89024 | daily_or_large_delta_move | quote and premium are good enough to test medium_dated_delta_hedge_check with explicit hedge PnL |
| btc_20260612_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 68.2410 | 4.65 | 2948.92 | 0.53 | 0.0440 | 81685 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| eth_20260610_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 51.7436 | 4.35 | 73.29 | 0.56 | 0.0471 | 37452 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| btc_20260611_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 51.3651 | 4.15 | 2632.21 | 0.56 | 0.0494 | 78703 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| eth_20260612_short_dated_delta_hedge_check | quote_only_hedge_watch | 50.6774 | 6.05 | 101.89 | 0.53 | 0.0336 | 17728 | 4h_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| btc_20260626_medium_dated_delta_hedge_check | quote_only_hedge_watch | 50.4007 | 8.15 | 5171.66 | 0.43 | 0.0248 | 1552 | daily_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| btc_20260610_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 49.0872 | 3.45 | 2187.87 | 0.58 | 0.0597 | 56666 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |

## Interpretation

Long-vol alpha is only useful if the option quote, premium-at-risk, delta hedge path, exit bid, and margin treatment are explicit. Rows here identify which structures are ready for that paper hedge check.
