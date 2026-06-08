# Current Policy Context Frontier

This summarizes observation/action/reward records by context. It is a prioritization board for alpha exploration, not a trained policy.

| context | decision | records | repeat | hit | mean | repeat mean | worst | best | score | next step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| repeat_execution | expand_context_now | 8 | 5 | 1.000 | 234.12 | 212.46 | 111.34 | 355.82 | 237.38 | open more repeat_execution paper labels across fresh assets, venues, and failure regimes |
| news_event | expand_context_now | 12 | 6 | 1.000 | 255.13 | 149.34 | 14.87 | 516.65 | 209.92 | open more news_event paper labels across fresh assets, venues, and failure regimes |
| volume_price_dislocation | expand_context_now | 36 | 14 | 1.000 | 166.88 | 107.84 | 14.87 | 538.39 | 171.03 | open more volume_price_dislocation paper labels across fresh assets, venues, and failure regimes |
| liquidation_intensity | expand_context_now | 2 | 2 | 1.000 | 239.79 | 239.79 | 221.18 | 258.41 | 169.89 | open more liquidation_intensity paper labels across fresh assets, venues, and failure regimes |
| microstructure_flow | expand_context_now | 44 | 20 | 0.909 | 137.47 | 115.44 | -106.80 | 659.11 | 148.63 | open more microstructure_flow paper labels across fresh assets, venues, and failure regimes |
| stablecoin_migration | expand_context_now | 4 | 2 | 1.000 | 182.76 | 143.92 | 107.88 | 263.22 | 130.27 | open more stablecoin_migration paper labels across fresh assets, venues, and failure regimes |
| event_crypto_hedge | expand_context_now | 28 | 14 | 0.893 | 70.32 | 63.14 | -188.23 | 179.96 | 100.48 | open more event_crypto_hedge paper labels across fresh assets, venues, and failure regimes |
| wallet_entity_flow | expand_with_failure_split | 8 | 4 | 0.875 | 56.31 | 91.48 | -546.86 | 346.42 | 68.20 | open more wallet_entity_flow labels, but split the failure regime before increasing size or confidence |
| sector_rotation | watch_context | 2 | 1 | 1.000 | 260.86 | 175.30 | 175.30 | 346.42 | 125.82 | keep sector_rotation on watch until repeat sample count exceeds 1 |
| event_pressure | watch_context | 2 | 1 | 1.000 | 163.17 | 97.91 | 97.91 | 228.43 | 90.76 | keep event_pressure on watch until repeat sample count exceeds 1 |
| basis_term_structure | watch_context | 1 | 0 | 0.000 | -96.97 | 0.00 | -96.97 | -96.97 | -33.44 | keep basis_term_structure on watch until repeat sample count exceeds 0 |
| execution_edge | shrink_or_rework_context | 10 | 2 | 0.400 | -0.57 | 107.88 | -546.86 | 179.96 | 6.04 | stop blind execution_edge expansion; isolate why repeat_mean=107.88 and worst=-546.86 |
| options_volatility | shrink_or_rework_context | 6 | 1 | 0.333 | -69.40 | 29.18 | -170.17 | 88.69 | -10.52 | stop blind options_volatility expansion; isolate why repeat_mean=29.18 and worst=-170.17 |
| unclassified | shrink_or_rework_context | 10 | 2 | 0.000 | -157.86 | -159.05 | -357.74 | 0.00 | -113.98 | stop blind unclassified expansion; isolate why repeat_mean=-159.05 and worst=-357.74 |
| intraday_derivatives | shrink_or_rework_context | 24 | 10 | 0.000 | -268.48 | -308.58 | -369.46 | -157.73 | -261.78 | stop blind intraday_derivatives expansion; isolate why repeat_mean=-308.58 and worst=-369.46 |
| protocol_fee | shrink_or_rework_context | 6 | 2 | 0.167 | -396.85 | -566.67 | -586.47 | 73.96 | -306.03 | stop blind protocol_fee expansion; isolate why repeat_mean=-566.67 and worst=-586.47 |
| token_unlock | shrink_or_rework_context | 4 | 2 | 0.000 | -552.19 | -571.22 | -586.47 | -525.12 | -351.78 | stop blind token_unlock expansion; isolate why repeat_mean=-571.22 and worst=-586.47 |

## Interpretation

`expand_context_now` means repeat samples are positive enough to open more labels. `expand_with_failure_split` means repeat samples are positive, but the worst loss is large enough that failure regimes must be separated before increasing confidence. `collect_repeat_samples` means the context is promising but under-tested. `shrink_or_rework_context` means the current paper evidence is net negative and should not receive more blind expansion.
