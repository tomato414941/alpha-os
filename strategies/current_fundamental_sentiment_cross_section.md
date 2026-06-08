# Current Fundamental Sentiment Cross Section

This is a first cross-sectional feature table from existing fundamental, sentiment, sector, and funding probes. It ranks research candidates; it is not a rebalance rule.

| symbol | decision | side | total | fundamental | sentiment | sector | funding | sources | conflict | next probe |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HMSTR | cross_section_watchlist | long_bias | 131.7361 | 0.0000 | 0.0000 | 100.8305 | 14.9055 | 2 | none | collect one more independent HMSTR feature before cross-section labeling |
| ZEC | cross_section_watchlist | long_bias | 117.1355 | 0.0000 | 39.5644 | 49.6391 | 3.9320 | 3 | none | collect one more independent ZEC feature before cross-section labeling |
| WLD | cross_section_watchlist | long_bias | 92.3327 | 0.0000 | 61.6500 | -6.6827 | 0.0000 | 3 | none | collect one more independent WLD feature before cross-section labeling |
| PENGU | cross_section_watchlist | long_bias | 82.7488 | 0.0000 | 65.4266 | 0.0000 | 1.3222 | 2 | none | collect one more independent PENGU feature before cross-section labeling |
| CRV | cross_section_watchlist | long_bias | 70.6816 | 54.3623 | 0.0000 | 0.0000 | 0.3193 | 2 | none | collect one more independent CRV feature before cross-section labeling |
| TAO | insufficient_cross_section_context | long_bias | 68.6005 | 0.0000 | 51.1703 | 0.0000 | 1.4301 | 2 | none | keep TAO as context until another feature source appears |
| DEUS | insufficient_cross_section_context | long_bias | 65.0000 | 0.0000 | 68.1243 | 0.0000 | 0.0000 | 1 | none | keep DEUS as context until another feature source appears |
| H | insufficient_cross_section_context | long_bias | 65.0000 | 0.0000 | 67.5000 | 0.0000 | 0.0000 | 1 | none | keep H as context until another feature source appears |
| HOME | insufficient_cross_section_context | long_bias | 65.0000 | 0.0000 | 0.0000 | 227.3889 | 0.0000 | 1 | none | keep HOME as context until another feature source appears |
| PIPPIN | insufficient_cross_section_context | long_bias | 65.0000 | 0.0000 | 61.8056 | 0.0000 | 0.0000 | 1 | none | keep PIPPIN as context until another feature source appears |
| JUP | insufficient_cross_section_context | long_bias | 62.8310 | 44.8386 | 0.0000 | 0.0000 | 1.9924 | 2 | none | keep JUP as context until another feature source appears |
| AAVE | insufficient_cross_section_context | long_bias | 58.9968 | 40.3859 | 0.0000 | 0.0000 | 2.6108 | 2 | none | keep AAVE as context until another feature source appears |
| PENDLE | insufficient_cross_section_context | long_bias | 58.3255 | 42.1298 | 0.0000 | 0.0000 | 0.1957 | 2 | none | keep PENDLE as context until another feature source appears |
| MORPHO | insufficient_cross_section_context | long_bias | 57.8823 | 41.6586 | 0.0000 | 0.0000 | 0.2237 | 2 | none | keep MORPHO as context until another feature source appears |
| UNI | insufficient_cross_section_context | short_bias | 50.5317 | -33.0933 | 0.0000 | 0.0000 | 1.4384 | 2 | none | keep UNI as context until another feature source appears |
| LDO | insufficient_cross_section_context | short_bias | 50.0154 | 0.0000 | 0.0000 | -34.0154 | 0.0000 | 2 | none | keep LDO as context until another feature source appears |
| SOL | insufficient_cross_section_context | short_bias | 49.3503 | -32.4997 | 0.0000 | 0.0000 | 0.8506 | 2 | none | keep SOL as context until another feature source appears |
| ETH | insufficient_cross_section_context | long_bias | 49.1737 | 0.0000 | 33.0641 | 0.0000 | 0.1096 | 2 | none | keep ETH as context until another feature source appears |
| HYPE | split_conflicting_features_before_label | long_bias | 45.3261 | 0.0000 | 31.5718 | -14.7544 | 0.0000 | 3 | long_short_feature_conflict | split HYPE features by sign before any cross-section label |
| ENS | insufficient_cross_section_context | short_bias | 44.1245 | 0.0000 | 0.0000 | -26.6827 | 1.4419 | 2 | none | keep ENS as context until another feature source appears |
| STRK | insufficient_cross_section_context | short_bias | 43.4043 | 0.0000 | 0.0000 | -27.2002 | 0.2041 | 2 | none | keep STRK as context until another feature source appears |
| LINK | insufficient_cross_section_context | long_bias | 42.2676 | 0.0000 | 0.0000 | 24.4698 | 1.7978 | 2 | none | keep LINK as context until another feature source appears |
| NEAR | split_conflicting_features_before_label | long_bias | 41.7692 | 0.0000 | 31.2255 | -11.5023 | 0.0414 | 3 | long_short_feature_conflict | split NEAR features by sign before any cross-section label |
| BEAT | insufficient_cross_section_context | long_bias | 39.5683 | 0.0000 | 5.4000 | 18.1683 | 0.0000 | 2 | none | keep BEAT as context until another feature source appears |
| RENDER | insufficient_cross_section_context | long_bias | 35.9557 | 0.0000 | 0.0000 | 19.0247 | 0.9310 | 2 | none | keep RENDER as context until another feature source appears |

## Interpretation

Rows with multiple sources are better cross-section candidates than single-lane rows. `split_conflicting_features_before_label` means the feature signs disagree and should not be collapsed into one trade. The next step is leakage-safe forward labeling by rebalance timestamp.
