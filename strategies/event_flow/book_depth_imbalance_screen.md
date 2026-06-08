# Book Depth Imbalance Screen

This checks whether futures bookDepth liquidity imbalance has a simple next-1m return edge. It is a data-path and diagnostic check, not a deployable strategy.

| feature | bucket | count | mean next return | hit rate |
| --- | --- | ---: | ---: | ---: |
| imbalance_1pct | bottom_20 | 1727 | 0.0000645554 | 0.514765 |
| imbalance_1pct | middle_60 | 5180 | 0.0000247872 | 0.511583 |
| imbalance_1pct | top_20 | 1727 | 0.0000396469 | 0.503185 |
| imbalance_5pct | bottom_20 | 1727 | 0.0000356761 | 0.491604 |
| imbalance_5pct | middle_60 | 5180 | 0.0000253247 | 0.512934 |
| imbalance_5pct | top_20 | 1727 | 0.0000669140 | 0.522293 |
