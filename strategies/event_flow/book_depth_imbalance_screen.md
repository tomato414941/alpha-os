# Book Depth Imbalance Screen

This checks whether futures bookDepth liquidity imbalance has a simple next-1m return edge. It is a data-path and diagnostic check, not a deployable strategy.

| feature | bucket | count | mean next return | hit rate |
| --- | --- | ---: | ---: | ---: |
| imbalance_1pct | bottom_20 | 39205 | -0.0000446218 | 0.460502 |
| imbalance_1pct | middle_60 | 117611 | -0.0000176463 | 0.468715 |
| imbalance_1pct | top_20 | 39204 | 0.0000090194 | 0.491098 |
| imbalance_5pct | bottom_20 | 39205 | -0.0000487032 | 0.467925 |
| imbalance_5pct | middle_60 | 117611 | -0.0000164358 | 0.468060 |
| imbalance_5pct | top_20 | 39204 | 0.0000094696 | 0.485639 |
