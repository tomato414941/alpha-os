# Current Fundamental Sentiment Cross Section

This is a first cross-sectional feature table from existing fundamental, sentiment, sector, and funding probes. It ranks research candidates; it is not a rebalance rule.

| symbol | decision | side | total | fundamental | sentiment | sector | funding | sources | conflict | next probe |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ZEC | cross_section_label_priority | long_bias | 121.6880 | 0.0000 | 39.6936 | 51.8749 | 6.1195 | 3 | none | label ZEC long_bias cross-section row at the next rebalance timestamp |
| JUP | cross_section_watchlist | long_bias | 88.2518 | 44.7857 | 0.0000 | 16.8867 | 2.5795 | 3 | none | collect one more independent JUP feature before cross-section labeling |
| PENGU | cross_section_watchlist | long_bias | 81.5908 | 0.0000 | 65.0083 | 0.0000 | 0.5825 | 2 | none | collect one more independent PENGU feature before cross-section labeling |
| HYPE | split_conflicting_features_before_label | long_bias | 81.5633 | 0.0000 | 70.2102 | -11.5763 | 0.7768 | 3 | long_short_feature_conflict | split HYPE features by sign before any cross-section label |
| BEAT | cross_section_watchlist | long_bias | 79.4437 | 0.0000 | 62.4012 | 1.0425 | 0.0000 | 2 | none | collect one more independent BEAT feature before cross-section labeling |
| TAO | cross_section_watchlist | long_bias | 73.0153 | 0.0000 | 54.5150 | 0.0000 | 2.5003 | 2 | none | collect one more independent TAO feature before cross-section labeling |
| CRV | cross_section_watchlist | long_bias | 71.0865 | 54.2616 | 0.0000 | 0.0000 | 0.8249 | 2 | none | collect one more independent CRV feature before cross-section labeling |
| SOL | split_conflicting_features_before_label | mixed_or_flat | 66.1978 | -32.4999 | 33.0624 | 0.0000 | 1.6355 | 3 | long_short_feature_conflict | split SOL features by sign before any cross-section label |
| ALLO | insufficient_cross_section_context | long_bias | 65.0000 | 0.0000 | 64.5750 | 0.0000 | 0.0000 | 1 | none | keep ALLO as context until another feature source appears |
| HOME | insufficient_cross_section_context | long_bias | 65.0000 | 0.0000 | 0.0000 | 203.3909 | 0.0000 | 1 | none | keep HOME as context until another feature source appears |
| LAYER | insufficient_cross_section_context | none | 63.5996 | 0.0000 | 0.0000 | 0.0000 | 55.5996 | 1 | none | keep LAYER as context until another feature source appears |
| PENDLE | insufficient_cross_section_context | long_bias | 59.0216 | 42.5044 | 0.0000 | 0.0000 | 0.5172 | 2 | none | keep PENDLE as context until another feature source appears |
| AAVE | insufficient_cross_section_context | long_bias | 58.5549 | 40.3859 | 0.0000 | 0.0000 | 2.1689 | 2 | none | keep AAVE as context until another feature source appears |
| BTC | insufficient_cross_section_context | long_bias | 54.8021 | 0.0000 | 35.8520 | 0.0000 | 2.9501 | 2 | none | keep BTC as context until another feature source appears |
| MORPHO | split_conflicting_features_before_label | long_bias | 52.1220 | 41.7899 | 0.0000 | -11.1201 | 0.2120 | 3 | long_short_feature_conflict | split MORPHO features by sign before any cross-section label |
| HMSTR | insufficient_cross_section_context | long_bias | 51.6585 | 0.0000 | 0.0000 | 29.5717 | 6.0868 | 2 | none | keep HMSTR as context until another feature source appears |
| LDO | insufficient_cross_section_context | short_bias | 50.0621 | 0.0000 | 0.0000 | -32.9795 | 1.0826 | 2 | none | keep LDO as context until another feature source appears |
| UNI | insufficient_cross_section_context | short_bias | 49.1013 | -33.1013 | 0.0000 | 0.0000 | 0.0000 | 2 | none | keep UNI as context until another feature source appears |
| VVV | insufficient_cross_section_context | short_bias | 42.7802 | 0.0000 | 6.6000 | -12.1802 | 0.0000 | 3 | none | keep VVV as context until another feature source appears |
| LIT | insufficient_cross_section_context | short_bias | 42.0204 | 0.0000 | 6.3000 | -11.7204 | 0.0000 | 3 | none | keep LIT as context until another feature source appears |
| STRK | insufficient_cross_section_context | short_bias | 41.7575 | 0.0000 | 0.0000 | -24.9646 | 0.7929 | 2 | none | keep STRK as context until another feature source appears |
| LINK | insufficient_cross_section_context | long_bias | 38.9584 | 0.0000 | 0.0000 | 21.6900 | 1.2684 | 2 | none | keep LINK as context until another feature source appears |
| JTO | insufficient_cross_section_context | long_bias | 38.3410 | 0.0000 | 0.0000 | 20.4391 | 1.9019 | 2 | none | keep JTO as context until another feature source appears |
| BIO | insufficient_cross_section_context | none | 38.3126 | 0.0000 | 0.0000 | 0.0000 | 30.3126 | 1 | none | keep BIO as context until another feature source appears |
| NEAR | insufficient_cross_section_context | long_bias | 35.1849 | 0.0000 | 0.0000 | 16.8850 | 2.2999 | 2 | none | keep NEAR as context until another feature source appears |

## Interpretation

Rows with multiple sources are better cross-section candidates than single-lane rows. `split_conflicting_features_before_label` means the feature signs disagree and should not be collapsed into one trade. The next step is leakage-safe forward labeling by rebalance timestamp.
