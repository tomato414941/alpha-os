# LOB Execution World Replay

This is a tiny RL-shaped replay over existing book-depth samples. The state is a train-side feature bucket, the action is hold/market/low-fee/maker-like execution, and the reward is next-1m directional return after explicit cost. Maker/internalized rows are optimistic full-fill diagnostics, not executable claims.

| feature | bucket | signal | execution action | fill assumption | train bps | test n | gross bps | cost bps | net bps | hit | decision | next step |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| imbalance_1pct | bottom_20 | paper_short | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.2782 | 21660 | 0.5957 | 0.50 | 0.0957 | 0.502401 | maker_fill_model_needed | build maker fill probability, queue position, cancel, and adverse-selection labels |
| taker_long_short_volume_ratio_5m | bottom_20 | paper_short | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.3293 | 13218 | 0.6613 | 0.50 | 0.1613 | 0.480557 | maker_tail_or_queue_research | treat as representation research until queue/fill and tail risk are measured |
| imbalance_1pct | bottom_20 | paper_short | hold | no_order | 0.2782 | 21660 | 0.5957 | 0.00 | 0.0000 | 0.502401 | hold_baseline | hold is the baseline action |
| open_interest_value_5m | top_20 | paper_short | hold | no_order | 0.2645 | 17208 | 0.0628 | 0.00 | 0.0000 | 0.500581 | hold_baseline | hold is the baseline action |
| imbalance_5pct | top_20 | paper_long | hold | no_order | 0.1073 | 9558 | 0.4707 | 0.00 | 0.0000 | 0.493304 | hold_baseline | hold is the baseline action |
| imbalance_5pct | bottom_20 | paper_short | hold | no_order | 0.4776 | 26579 | 0.4362 | 0.00 | 0.0000 | 0.492569 | hold_baseline | hold is the baseline action |
| mark_index_basis_1m | top_20 | paper_short | hold | no_order | 0.2301 | 16130 | -0.0563 | 0.00 | 0.0000 | 0.492498 | hold_baseline | hold is the baseline action |
| premium_index_1m | top_20 | paper_short | hold | no_order | 0.2192 | 19368 | 0.3717 | 0.00 | 0.0000 | 0.491429 | hold_baseline | hold is the baseline action |
| taker_long_short_volume_ratio_5m | bottom_20 | paper_short | hold | no_order | 0.3293 | 13218 | 0.6613 | 0.00 | 0.0000 | 0.480557 | hold_baseline | hold is the baseline action |
| top_trader_long_short_ratio_5m | top_20 | paper_short | hold | no_order | 0.2916 | 8604 | 0.0690 | 0.00 | 0.0000 | 0.477685 | hold_baseline | hold is the baseline action |
| mark_index_basis_1m | bottom_20 | paper_long | hold | no_order | 0.0887 | 28480 | -0.1302 | 0.00 | 0.0000 | 0.476826 | hold_baseline | hold is the baseline action |
| top_trader_long_short_ratio_5m | bottom_20 | paper_short | hold | no_order | 0.2532 | 50366 | -0.0613 | 0.00 | 0.0000 | 0.475718 | hold_baseline | hold is the baseline action |
| taker_long_short_volume_ratio_5m | top_20 | paper_short | hold | no_order | 0.1472 | 15984 | -0.2274 | 0.00 | 0.0000 | 0.474975 | hold_baseline | hold is the baseline action |
| account_long_short_ratio_5m | top_20 | paper_short | hold | no_order | 0.2587 | 8604 | 0.0808 | 0.00 | 0.0000 | 0.470944 | hold_baseline | hold is the baseline action |
| account_long_short_ratio_5m | bottom_20 | paper_long | hold | no_order | 0.1007 | 17508 | -0.1701 | 0.00 | 0.0000 | 0.466187 | hold_baseline | hold is the baseline action |
| premium_index_1m | bottom_20 | paper_short | hold | no_order | 0.3238 | 28560 | -0.0055 | 0.00 | 0.0000 | 0.465406 | hold_baseline | hold is the baseline action |
| open_interest_value_5m | bottom_20 | paper_short | hold | no_order | 0.0831 | 24662 | 0.0987 | 0.00 | 0.0000 | 0.459330 | hold_baseline | hold is the baseline action |
| imbalance_1pct | top_20 | paper_short | hold | no_order | 0.0040 | 11807 | -0.6043 | 0.00 | 0.0000 | 0.456001 | hold_baseline | hold is the baseline action |
| imbalance_5pct | top_20 | paper_long | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.1073 | 9558 | 0.4707 | 0.50 | -0.0293 | 0.493304 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| imbalance_5pct | bottom_20 | paper_short | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.4776 | 26579 | 0.4362 | 0.50 | -0.0638 | 0.492569 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| premium_index_1m | top_20 | paper_short | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.2192 | 19368 | 0.3717 | 0.50 | -0.1283 | 0.491429 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| open_interest_value_5m | bottom_20 | paper_short | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.0831 | 24662 | 0.0987 | 0.50 | -0.4013 | 0.459330 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| account_long_short_ratio_5m | top_20 | paper_short | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.2587 | 8604 | 0.0808 | 0.50 | -0.4192 | 0.470944 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| top_trader_long_short_ratio_5m | top_20 | paper_short | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.2916 | 8604 | 0.0690 | 0.50 | -0.4310 | 0.477685 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| open_interest_value_5m | top_20 | paper_short | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.2645 | 17208 | 0.0628 | 0.50 | -0.4372 | 0.500581 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| premium_index_1m | bottom_20 | paper_short | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.3238 | 28560 | -0.0055 | 0.50 | -0.5055 | 0.465406 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| mark_index_basis_1m | top_20 | paper_short | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.2301 | 16130 | -0.0563 | 0.50 | -0.5563 | 0.492498 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| top_trader_long_short_ratio_5m | bottom_20 | paper_short | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.2532 | 50366 | -0.0613 | 0.50 | -0.5613 | 0.475718 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| mark_index_basis_1m | bottom_20 | paper_long | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.0887 | 28480 | -0.1302 | 0.50 | -0.6302 | 0.476826 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| account_long_short_ratio_5m | bottom_20 | paper_long | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.1007 | 17508 | -0.1701 | 0.50 | -0.6701 | 0.466187 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| taker_long_short_volume_ratio_5m | top_20 | paper_short | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.1472 | 15984 | -0.2274 | 0.50 | -0.7274 | 0.474975 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| imbalance_1pct | top_20 | paper_short | maker_or_internalized_limit | optimistic_full_fill_not_queue_verified | 0.0040 | 11807 | -0.6043 | 0.50 | -1.1043 | 0.456001 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| taker_long_short_volume_ratio_5m | bottom_20 | paper_short | low_fee_cross | full_fill_low_fee_or_internalized | 0.3293 | 13218 | 0.6613 | 2.00 | -1.3387 | 0.480557 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| imbalance_1pct | bottom_20 | paper_short | low_fee_cross | full_fill_low_fee_or_internalized | 0.2782 | 21660 | 0.5957 | 2.00 | -1.4043 | 0.502401 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| imbalance_5pct | top_20 | paper_long | low_fee_cross | full_fill_low_fee_or_internalized | 0.1073 | 9558 | 0.4707 | 2.00 | -1.5293 | 0.493304 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| imbalance_5pct | bottom_20 | paper_short | low_fee_cross | full_fill_low_fee_or_internalized | 0.4776 | 26579 | 0.4362 | 2.00 | -1.5638 | 0.492569 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| premium_index_1m | top_20 | paper_short | low_fee_cross | full_fill_low_fee_or_internalized | 0.2192 | 19368 | 0.3717 | 2.00 | -1.6283 | 0.491429 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| open_interest_value_5m | bottom_20 | paper_short | low_fee_cross | full_fill_low_fee_or_internalized | 0.0831 | 24662 | 0.0987 | 2.00 | -1.9013 | 0.459330 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| account_long_short_ratio_5m | top_20 | paper_short | low_fee_cross | full_fill_low_fee_or_internalized | 0.2587 | 8604 | 0.0808 | 2.00 | -1.9192 | 0.470944 | worse_than_hold | do not act under this execution action; hold beats the replay reward |
| top_trader_long_short_ratio_5m | top_20 | paper_short | low_fee_cross | full_fill_low_fee_or_internalized | 0.2916 | 8604 | 0.0690 | 2.00 | -1.9310 | 0.477685 | worse_than_hold | do not act under this execution action; hold beats the replay reward |

## Summary

- hold_baseline: 16
- maker_fill_model_needed: 1
- maker_tail_or_queue_research: 1
- worse_than_hold: 46
- best non-hold action: imbalance_1pct/bottom_20/paper_short/maker_or_internalized_limit net=0.09571460bps hit=0.502401 decision=maker_fill_model_needed
