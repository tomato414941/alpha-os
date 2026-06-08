# Current Policy Context Frontier

This summarizes observation/action/reward records by context. It is a prioritization board for alpha exploration, not a trained policy.

| context | decision | records | repeat | hit | mean | repeat mean | worst | best | score | next step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| news_event | expand_with_failure_split | 26 | 8 | 0.615 | 228.54 | 345.53 | -206.05 | 703.22 | 282.05 | open more news_event labels, but split the failure regime before increasing size or confidence |
| sector_rotation | expand_context_now | 11 | 2 | 0.909 | 317.46 | 429.72 | -33.12 | 526.22 | 240.18 | open more sector_rotation paper labels across fresh assets, venues, and failure regimes |
| wallet_entity_flow | expand_with_failure_split | 20 | 11 | 0.700 | 112.21 | 104.71 | -329.55 | 703.22 | 104.11 | open more wallet_entity_flow labels, but split the failure regime before increasing size or confidence |
| liquidation_intensity | expand_context_now | 2 | 2 | 1.000 | 113.78 | 113.78 | 95.39 | 132.17 | 103.74 | open more liquidation_intensity paper labels across fresh assets, venues, and failure regimes |
| repeat_execution | expand_context_now | 9 | 6 | 0.778 | 130.57 | 53.89 | -120.91 | 283.91 | 96.47 | open more repeat_execution paper labels across fresh assets, venues, and failure regimes |
| execution_edge | expand_with_failure_split | 11 | 2 | 0.818 | 86.71 | 104.08 | -329.55 | 283.91 | 60.75 | open more execution_edge labels, but split the failure regime before increasing size or confidence |
| stablecoin_migration | watch_context | 7 | 4 | 0.571 | 115.59 | 68.13 | -3.85 | 281.13 | 100.48 | keep stablecoin_migration on watch until repeat sample count exceeds 4 |
| volume_price_dislocation | shrink_or_rework_context | 43 | 32 | 0.442 | -1.06 | -2.87 | -206.05 | 299.97 | 16.92 | stop blind volume_price_dislocation expansion; isolate why repeat_mean=-2.87 and worst=-206.05 |
| event_crypto_hedge | shrink_or_rework_context | 39 | 23 | 0.436 | -3.18 | -10.87 | -184.31 | 176.13 | 10.74 | stop blind event_crypto_hedge expansion; isolate why repeat_mean=-10.87 and worst=-184.31 |
| basis_term_structure | shrink_or_rework_context | 4 | 0 | 0.500 | -4.20 | 0.00 | -84.95 | 76.92 | 6.46 | stop blind basis_term_structure expansion; isolate why repeat_mean=0.00 and worst=-84.95 |
| microstructure_flow | shrink_or_rework_context | 45 | 40 | 0.511 | 15.34 | -10.61 | -314.55 | 743.48 | 6.35 | stop blind microstructure_flow expansion; isolate why repeat_mean=-10.61 and worst=-314.55 |
| event_pressure | shrink_or_rework_context | 9 | 2 | 0.333 | 26.63 | -64.34 | -206.05 | 333.21 | -17.81 | stop blind event_pressure expansion; isolate why repeat_mean=-64.34 and worst=-206.05 |
| options_volatility | shrink_or_rework_context | 6 | 1 | 0.167 | -48.16 | -47.81 | -110.85 | 2.92 | -22.03 | stop blind options_volatility expansion; isolate why repeat_mean=-47.81 and worst=-110.85 |
| unclassified | shrink_or_rework_context | 15 | 2 | 0.267 | -130.48 | -137.30 | -712.90 | 333.21 | -119.20 | stop blind unclassified expansion; isolate why repeat_mean=-137.30 and worst=-712.90 |
| protocol_fee | shrink_or_rework_context | 8 | 2 | 0.375 | -169.26 | -348.94 | -368.34 | 75.96 | -160.61 | stop blind protocol_fee expansion; isolate why repeat_mean=-348.94 and worst=-368.34 |
| token_unlock | shrink_or_rework_context | 3 | 2 | 0.000 | -338.35 | -353.40 | -368.34 | -308.25 | -217.11 | stop blind token_unlock expansion; isolate why repeat_mean=-353.40 and worst=-368.34 |
| intraday_derivatives | shrink_or_rework_context | 24 | 10 | 0.042 | -209.00 | -263.96 | -353.44 | 8.93 | -219.60 | stop blind intraday_derivatives expansion; isolate why repeat_mean=-263.96 and worst=-353.44 |

## Interpretation

`expand_context_now` means repeat samples are positive enough to open more labels. `expand_with_failure_split` means repeat samples are positive, but the worst loss is large enough that failure regimes must be separated before increasing confidence. `collect_repeat_samples` means the context is promising but under-tested. `shrink_or_rework_context` means the current paper evidence is net negative and should not receive more blind expansion.
