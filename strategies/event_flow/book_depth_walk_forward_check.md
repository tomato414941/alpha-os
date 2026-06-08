# Book Depth Walk Forward Check

This uses train-side buckets to choose direction, skips a purge window, then checks test-side next-1m returns after an explicit round-trip cost. It is a diagnostic gate, not a deployable strategy.

| feature | bucket | action | train n | train bps | test n | gross bps | net bps | hit | decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| taker_long_short_volume_ratio_5m | bottom_20 | paper_short | 16191 | 0.3293 | 13218 | 0.6613 | -7.3387 | 0.480557 | gross_only_candidate |
| imbalance_1pct | bottom_20 | paper_short | 16183 | 0.2782 | 21660 | 0.5957 | -7.4043 | 0.502401 | gross_only_candidate |
| imbalance_5pct | top_20 | paper_long | 16182 | 0.1073 | 9558 | 0.4707 | -7.5293 | 0.493304 | gross_only_candidate |
| imbalance_5pct | bottom_20 | paper_short | 16183 | 0.4776 | 26579 | 0.4362 | -7.5638 | 0.492569 | gross_only_candidate |
| premium_index_1m | top_20 | paper_short | 16184 | 0.2192 | 19368 | 0.3717 | -7.6283 | 0.491429 | gross_only_candidate |
| open_interest_value_5m | bottom_20 | paper_short | 16184 | 0.0831 | 24662 | 0.0987 | -7.9013 | 0.459330 | gross_only_candidate |
| account_long_short_ratio_5m | top_20 | paper_short | 16187 | 0.2587 | 8604 | 0.0808 | -7.9192 | 0.470944 | gross_only_candidate |
| top_trader_long_short_ratio_5m | top_20 | paper_short | 16190 | 0.2916 | 8604 | 0.0690 | -7.9310 | 0.477685 | gross_only_candidate |
| open_interest_value_5m | top_20 | paper_short | 16190 | 0.2645 | 17208 | 0.0628 | -7.9372 | 0.500581 | gross_only_candidate |
| premium_index_1m | bottom_20 | paper_short | 16183 | 0.3238 | 28560 | -0.0055 | -8.0055 | 0.465406 | reject_after_walk_forward |
| mark_index_basis_1m | top_20 | paper_short | 16183 | 0.2301 | 16130 | -0.0563 | -8.0563 | 0.492498 | reject_after_walk_forward |
| top_trader_long_short_ratio_5m | bottom_20 | paper_short | 16187 | 0.2532 | 50366 | -0.0613 | -8.0613 | 0.475718 | reject_after_walk_forward |
| mark_index_basis_1m | bottom_20 | paper_long | 16183 | 0.0887 | 28480 | -0.1302 | -8.1302 | 0.476826 | reject_after_walk_forward |
| account_long_short_ratio_5m | bottom_20 | paper_long | 16185 | 0.1007 | 17508 | -0.1701 | -8.1701 | 0.466187 | reject_after_walk_forward |
| taker_long_short_volume_ratio_5m | top_20 | paper_short | 16182 | 0.1472 | 15984 | -0.2274 | -8.2274 | 0.474975 | reject_after_walk_forward |
| imbalance_1pct | top_20 | paper_short | 16182 | 0.0040 | 11807 | -0.6043 | -8.6043 | 0.456001 | reject_after_walk_forward |
