# Current Volatility Hedge Candidates

This turns option actionability rows into hedge-plan candidates. It is not a live options execution instruction.

| candidate | decision | score | max loss % | max loss USD | prem/rv move | spread | depth USD | hedge interval | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| btc_20260619_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 73.5592 | 6.20 | 3964.50 | 0.42 | 0.0328 | 227959 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| eth_20260619_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 70.3542 | 8.55 | 144.70 | 0.45 | 0.0237 | 83925 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| eth_20260626_medium_dated_delta_hedge_check | paper_delta_hedge_candidate | 67.8276 | 10.60 | 179.52 | 0.43 | 0.0287 | 34827 | daily_or_large_delta_move | quote and premium are good enough to test medium_dated_delta_hedge_check with explicit hedge PnL |
| btc_20260612_short_dated_delta_hedge_check | paper_delta_hedge_candidate | 67.0014 | 4.00 | 2557.27 | 0.46 | 0.0513 | 41939 | 4h_or_large_delta_move | quote and premium are good enough to test short_dated_delta_hedge_check with explicit hedge PnL |
| eth_20260611_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 52.2544 | 4.65 | 78.74 | 0.49 | 0.0440 | 36458 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| btc_20260611_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 50.3910 | 3.50 | 2238.35 | 0.47 | 0.0588 | 33128 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| eth_20260610_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 50.1828 | 3.65 | 61.81 | 0.48 | 0.0563 | 7169 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |
| eth_20260612_short_dated_delta_hedge_check | quote_only_hedge_watch | 50.1182 | 5.45 | 92.23 | 0.49 | 0.0469 | 10699 | 4h_or_large_delta_move | quote exists, but depth or mechanics are not strong enough for hedge promotion |
| btc_20260610_expiry_gamma_scalp_check | expiry_gamma_hedge_watch | 48.6156 | 2.70 | 1726.73 | 0.46 | 0.0769 | 52320 | 2h_or_large_delta_move | near-expiry gamma may be useful, but hedge timing dominates the result |

## Interpretation

Long-vol alpha is only useful if the option quote, premium-at-risk, delta hedge path, exit bid, and margin treatment are explicit. Rows here identify which structures are ready for that paper hedge check.
