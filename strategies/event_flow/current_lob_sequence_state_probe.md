# LOB Sequence State Probe

This turns book-depth snapshots into rolling state features before evaluating the next-1m label. It is a representation probe: positive zero-cost rows are not alpha unless they survive an execution mode.

| feature | bucket | signal | mode | train n | train bps | test n | gross bps | cost bps | net bps | hit | decision | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ask_liquidity_change_5 | top_20 | paper_short | maker_or_internalized_limit | 16152 | 0.1210 | 21341 | 0.5475 | 0.50 | 0.0475 | 0.494869 | maker_sequence_tail_candidate | inspect tail dependence and adverse selection before trusting maker execution |
| ask_liquidity_change_5 | top_20 | paper_short | zero_cost_representation | 16152 | 0.1210 | 21341 | 0.5475 | 0.00 | 0.5475 | 0.494869 | representation_only | keep as a feature for a model; it does not survive execution costs |
| basis_delta_5 | top_20 | paper_long | zero_cost_representation | 16152 | 0.0407 | 22031 | 0.4159 | 0.00 | 0.4159 | 0.473651 | representation_only | keep as a feature for a model; it does not survive execution costs |
| shallow_deep_imbalance_gap | bottom_20 | paper_short | zero_cost_representation | 16153 | 0.1123 | 19180 | 0.4014 | 0.00 | 0.4014 | 0.498436 | representation_only | keep as a feature for a model; it does not survive execution costs |
| imbalance_5pct_persistence_5 | bottom_20 | paper_short | zero_cost_representation | 16153 | 0.4363 | 24578 | 0.3677 | 0.00 | 0.3677 | 0.482261 | representation_only | keep as a feature for a model; it does not survive execution costs |
| imbalance_1pct_persistence_5 | bottom_20 | paper_short | zero_cost_representation | 16153 | 0.3128 | 20963 | 0.2881 | 0.00 | 0.2881 | 0.485951 | representation_only | keep as a feature for a model; it does not survive execution costs |
| taker_pressure_delta_5 | bottom_20 | paper_short | zero_cost_representation | 16153 | 0.5774 | 15920 | 0.2861 | 0.00 | 0.2861 | 0.477952 | representation_only | keep as a feature for a model; it does not survive execution costs |
| premium_delta_5 | bottom_20 | paper_short | zero_cost_representation | 16153 | 0.2884 | 23253 | 0.2627 | 0.00 | 0.2627 | 0.475853 | representation_only | keep as a feature for a model; it does not survive execution costs |
| imbalance_5pct_persistence_5 | top_20 | paper_long | zero_cost_representation | 16152 | 0.0695 | 10018 | 0.2440 | 0.00 | 0.2440 | 0.488022 | representation_only | keep as a feature for a model; it does not survive execution costs |
| basis_delta_5 | top_20 | paper_long | maker_or_internalized_limit | 16152 | 0.0407 | 22031 | 0.4159 | 0.50 | -0.0841 | 0.473651 | reject_after_cost | reject this rolling state under the current execution cost |
| shallow_deep_imbalance_gap | bottom_20 | paper_short | maker_or_internalized_limit | 16153 | 0.1123 | 19180 | 0.4014 | 0.50 | -0.0986 | 0.498436 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_1pct_persistence_5 | top_20 | paper_short | zero_cost_representation | 16152 | 0.0225 | 12138 | -0.1148 | 0.00 | -0.1148 | 0.476520 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_5pct_persistence_5 | bottom_20 | paper_short | maker_or_internalized_limit | 16153 | 0.4363 | 24578 | 0.3677 | 0.50 | -0.1323 | 0.482261 | reject_after_cost | reject this rolling state under the current execution cost |
| ask_liquidity_change_5 | bottom_20 | paper_short | zero_cost_representation | 16153 | 0.3509 | 21107 | -0.1688 | 0.00 | -0.1688 | 0.464538 | reject_after_cost | reject this rolling state under the current execution cost |
| premium_delta_5 | top_20 | paper_short | zero_cost_representation | 16152 | 0.0980 | 23395 | -0.1887 | 0.00 | -0.1887 | 0.476982 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_1pct_persistence_5 | bottom_20 | paper_short | maker_or_internalized_limit | 16153 | 0.3128 | 20963 | 0.2881 | 0.50 | -0.2119 | 0.485951 | reject_after_cost | reject this rolling state under the current execution cost |
| taker_pressure_delta_5 | bottom_20 | paper_short | maker_or_internalized_limit | 16153 | 0.5774 | 15920 | 0.2861 | 0.50 | -0.2139 | 0.477952 | reject_after_cost | reject this rolling state under the current execution cost |
| premium_delta_5 | bottom_20 | paper_short | maker_or_internalized_limit | 16153 | 0.2884 | 23253 | 0.2627 | 0.50 | -0.2373 | 0.475853 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_1pct_delta_5 | top_20 | paper_short | zero_cost_representation | 16152 | 0.3089 | 21700 | -0.2506 | 0.00 | -0.2506 | 0.463088 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_5pct_persistence_5 | top_20 | paper_long | maker_or_internalized_limit | 16152 | 0.0695 | 10018 | 0.2440 | 0.50 | -0.2560 | 0.488022 | reject_after_cost | reject this rolling state under the current execution cost |
| taker_pressure_delta_5 | top_20 | paper_short | zero_cost_representation | 16152 | 0.0465 | 15952 | -0.2835 | 0.00 | -0.2835 | 0.470411 | reject_after_cost | reject this rolling state under the current execution cost |
| bid_liquidity_change_5 | top_20 | paper_short | zero_cost_representation | 16152 | 0.0269 | 21145 | -0.3193 | 0.00 | -0.3193 | 0.463561 | reject_after_cost | reject this rolling state under the current execution cost |
| bid_liquidity_change_5 | bottom_20 | paper_long | zero_cost_representation | 16153 | 0.1471 | 21099 | -0.3623 | 0.00 | -0.3623 | 0.465899 | reject_after_cost | reject this rolling state under the current execution cost |
| oi_value_change_15 | top_20 | paper_short | zero_cost_representation | 16152 | 0.1850 | 20224 | -0.4213 | 0.00 | -0.4213 | 0.482595 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_5pct_delta_5 | top_20 | paper_short | zero_cost_representation | 16152 | 0.2847 | 21691 | -0.4717 | 0.00 | -0.4717 | 0.462081 | reject_after_cost | reject this rolling state under the current execution cost |
| shallow_deep_imbalance_gap | top_20 | paper_short | zero_cost_representation | 16152 | 0.1615 | 18115 | -0.5332 | 0.00 | -0.5332 | 0.451615 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_1pct_delta_5 | bottom_20 | paper_long | zero_cost_representation | 16153 | 0.0636 | 21902 | -0.6063 | 0.00 | -0.6063 | 0.458132 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_1pct_persistence_5 | top_20 | paper_short | maker_or_internalized_limit | 16152 | 0.0225 | 12138 | -0.1148 | 0.50 | -0.6148 | 0.476520 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_5pct_delta_5 | bottom_20 | paper_long | zero_cost_representation | 16153 | 0.0953 | 21203 | -0.6247 | 0.00 | -0.6247 | 0.455549 | reject_after_cost | reject this rolling state under the current execution cost |
| ask_liquidity_change_5 | bottom_20 | paper_short | maker_or_internalized_limit | 16153 | 0.3509 | 21107 | -0.1688 | 0.50 | -0.6688 | 0.464538 | reject_after_cost | reject this rolling state under the current execution cost |
| premium_delta_5 | top_20 | paper_short | maker_or_internalized_limit | 16152 | 0.0980 | 23395 | -0.1887 | 0.50 | -0.6887 | 0.476982 | reject_after_cost | reject this rolling state under the current execution cost |
| basis_delta_5 | bottom_20 | paper_long | zero_cost_representation | 16153 | 0.0187 | 22031 | -0.6975 | 0.00 | -0.6975 | 0.465254 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_1pct_delta_5 | top_20 | paper_short | maker_or_internalized_limit | 16152 | 0.3089 | 21700 | -0.2506 | 0.50 | -0.7506 | 0.463088 | reject_after_cost | reject this rolling state under the current execution cost |
| taker_pressure_delta_5 | top_20 | paper_short | maker_or_internalized_limit | 16152 | 0.0465 | 15952 | -0.2835 | 0.50 | -0.7835 | 0.470411 | reject_after_cost | reject this rolling state under the current execution cost |
| bid_liquidity_change_5 | top_20 | paper_short | maker_or_internalized_limit | 16152 | 0.0269 | 21145 | -0.3193 | 0.50 | -0.8193 | 0.463561 | reject_after_cost | reject this rolling state under the current execution cost |
| bid_liquidity_change_5 | bottom_20 | paper_long | maker_or_internalized_limit | 16153 | 0.1471 | 21099 | -0.3623 | 0.50 | -0.8623 | 0.465899 | reject_after_cost | reject this rolling state under the current execution cost |
| oi_value_change_15 | top_20 | paper_short | maker_or_internalized_limit | 16152 | 0.1850 | 20224 | -0.4213 | 0.50 | -0.9213 | 0.482595 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_5pct_delta_5 | top_20 | paper_short | maker_or_internalized_limit | 16152 | 0.2847 | 21691 | -0.4717 | 0.50 | -0.9717 | 0.462081 | reject_after_cost | reject this rolling state under the current execution cost |
| shallow_deep_imbalance_gap | top_20 | paper_short | maker_or_internalized_limit | 16152 | 0.1615 | 18115 | -0.5332 | 0.50 | -1.0332 | 0.451615 | reject_after_cost | reject this rolling state under the current execution cost |
| oi_value_change_15 | bottom_20 | paper_long | zero_cost_representation | 16153 | 0.0388 | 19841 | -1.0414 | 0.00 | -1.0414 | 0.462275 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_1pct_delta_5 | bottom_20 | paper_long | maker_or_internalized_limit | 16153 | 0.0636 | 21902 | -0.6063 | 0.50 | -1.1063 | 0.458132 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_5pct_delta_5 | bottom_20 | paper_long | maker_or_internalized_limit | 16153 | 0.0953 | 21203 | -0.6247 | 0.50 | -1.1247 | 0.455549 | reject_after_cost | reject this rolling state under the current execution cost |
| basis_delta_5 | bottom_20 | paper_long | maker_or_internalized_limit | 16153 | 0.0187 | 22031 | -0.6975 | 0.50 | -1.1975 | 0.465254 | reject_after_cost | reject this rolling state under the current execution cost |
| ask_liquidity_change_5 | top_20 | paper_short | low_fee_cross | 16152 | 0.1210 | 21341 | 0.5475 | 2.00 | -1.4525 | 0.494869 | reject_after_cost | reject this rolling state under the current execution cost |
| oi_value_change_15 | bottom_20 | paper_long | maker_or_internalized_limit | 16153 | 0.0388 | 19841 | -1.0414 | 0.50 | -1.5414 | 0.462275 | reject_after_cost | reject this rolling state under the current execution cost |
| basis_delta_5 | top_20 | paper_long | low_fee_cross | 16152 | 0.0407 | 22031 | 0.4159 | 2.00 | -1.5841 | 0.473651 | reject_after_cost | reject this rolling state under the current execution cost |
| shallow_deep_imbalance_gap | bottom_20 | paper_short | low_fee_cross | 16153 | 0.1123 | 19180 | 0.4014 | 2.00 | -1.5986 | 0.498436 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_5pct_persistence_5 | bottom_20 | paper_short | low_fee_cross | 16153 | 0.4363 | 24578 | 0.3677 | 2.00 | -1.6323 | 0.482261 | reject_after_cost | reject this rolling state under the current execution cost |
| imbalance_1pct_persistence_5 | bottom_20 | paper_short | low_fee_cross | 16153 | 0.3128 | 20963 | 0.2881 | 2.00 | -1.7119 | 0.485951 | reject_after_cost | reject this rolling state under the current execution cost |
| taker_pressure_delta_5 | bottom_20 | paper_short | low_fee_cross | 16153 | 0.5774 | 15920 | 0.2861 | 2.00 | -1.7139 | 0.477952 | reject_after_cost | reject this rolling state under the current execution cost |

## Summary

- maker_sequence_tail_candidate: 1
- reject_after_cost: 79
- representation_only: 8
- best non-reject: ask_liquidity_change_5/top_20/paper_short/maker_or_internalized_limit net=0.04745233bps hit=0.494869 decision=maker_sequence_tail_candidate
