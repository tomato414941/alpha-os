# Book Depth Context Screen

This checks whether futures bookDepth liquidity imbalance and perp basis context have a simple next-1m return edge. It is a data-path and diagnostic check, not a deployable strategy.

| feature | bucket | count | mean next return | hit rate |
| --- | --- | ---: | ---: | ---: |
| imbalance_1pct | bottom_20 | 39205 | -0.0000446218 | 0.460502 |
| imbalance_1pct | middle_60 | 117611 | -0.0000176463 | 0.468715 |
| imbalance_1pct | top_20 | 39204 | 0.0000090194 | 0.491098 |
| imbalance_5pct | bottom_20 | 39205 | -0.0000487032 | 0.467925 |
| imbalance_5pct | middle_60 | 117611 | -0.0000164358 | 0.468060 |
| imbalance_5pct | top_20 | 39204 | 0.0000094696 | 0.485639 |
| premium_index_1m | bottom_20 | 39206 | -0.0000038795 | 0.477631 |
| premium_index_1m | middle_60 | 117607 | -0.0000144721 | 0.469853 |
| premium_index_1m | top_20 | 39207 | -0.0000412446 | 0.470554 |
| mark_index_basis_1m | bottom_20 | 39205 | 0.0000118134 | 0.477669 |
| mark_index_basis_1m | middle_60 | 117610 | -0.0000285522 | 0.468770 |
| mark_index_basis_1m | top_20 | 39205 | -0.0000147001 | 0.473766 |
