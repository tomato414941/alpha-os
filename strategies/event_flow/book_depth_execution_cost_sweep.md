# Book Depth Execution Cost Sweep

This re-prices book-depth walk-forward rows across execution-cost assumptions. It separates signals that are dead for taker execution from signals that might only survive with low-fee, maker, or internalized execution. It is not a trade instruction.

| feature | bucket | action | mode | gross bps | cost bps | net bps | hit | status | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| imbalance_1pct | bottom_20 | paper_short | maker_or_internalized | 0.59571460 | 0.50 | 0.0957 | 0.502401 | maker_or_internalized_candidate | test maker fill probability, queue position, adverse selection, and cancellation rules |
| taker_long_short_volume_ratio_5m | bottom_20 | paper_short | maker_or_internalized | 0.66130790 | 0.50 | 0.1613 | 0.480557 | maker_only_low_hit_rate | do not trade directionally; inspect whether payoff tail or queue selection explains the weak hit rate |
| taker_long_short_volume_ratio_5m | bottom_20 | paper_short | zero_cost_diagnostic | 0.66130790 | 0.00 | 0.6613 | 0.480557 | zero_cost_only_signal | keep as representation-learning feature only; it is not execution-ready |
| imbalance_1pct | bottom_20 | paper_short | zero_cost_diagnostic | 0.59571460 | 0.00 | 0.5957 | 0.502401 | zero_cost_only_signal | keep as representation-learning feature only; it is not execution-ready |
| imbalance_5pct | top_20 | paper_long | zero_cost_diagnostic | 0.47073228 | 0.00 | 0.4707 | 0.493304 | zero_cost_only_signal | keep as representation-learning feature only; it is not execution-ready |
| imbalance_5pct | bottom_20 | paper_short | zero_cost_diagnostic | 0.43621187 | 0.00 | 0.4362 | 0.492569 | zero_cost_only_signal | keep as representation-learning feature only; it is not execution-ready |
| premium_index_1m | top_20 | paper_short | zero_cost_diagnostic | 0.37165082 | 0.00 | 0.3717 | 0.491429 | zero_cost_only_signal | keep as representation-learning feature only; it is not execution-ready |
| open_interest_value_5m | bottom_20 | paper_short | zero_cost_diagnostic | 0.09872878 | 0.00 | 0.0987 | 0.459330 | zero_cost_only_signal | keep as representation-learning feature only; it is not execution-ready |
| account_long_short_ratio_5m | top_20 | paper_short | zero_cost_diagnostic | 0.08075039 | 0.00 | 0.0808 | 0.470944 | zero_cost_only_signal | keep as representation-learning feature only; it is not execution-ready |
| top_trader_long_short_ratio_5m | top_20 | paper_short | zero_cost_diagnostic | 0.06896910 | 0.00 | 0.0690 | 0.477685 | zero_cost_only_signal | keep as representation-learning feature only; it is not execution-ready |
| open_interest_value_5m | top_20 | paper_short | zero_cost_diagnostic | 0.06276972 | 0.00 | 0.0628 | 0.500581 | zero_cost_only_signal | keep as representation-learning feature only; it is not execution-ready |
| imbalance_5pct | top_20 | paper_long | maker_or_internalized | 0.47073228 | 0.50 | -0.0293 | 0.493304 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| imbalance_5pct | bottom_20 | paper_short | maker_or_internalized | 0.43621187 | 0.50 | -0.0638 | 0.492569 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| premium_index_1m | top_20 | paper_short | maker_or_internalized | 0.37165082 | 0.50 | -0.1283 | 0.491429 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| open_interest_value_5m | bottom_20 | paper_short | maker_or_internalized | 0.09872878 | 0.50 | -0.4013 | 0.459330 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| account_long_short_ratio_5m | top_20 | paper_short | maker_or_internalized | 0.08075039 | 0.50 | -0.4192 | 0.470944 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| top_trader_long_short_ratio_5m | top_20 | paper_short | maker_or_internalized | 0.06896910 | 0.50 | -0.4310 | 0.477685 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| open_interest_value_5m | top_20 | paper_short | maker_or_internalized | 0.06276972 | 0.50 | -0.4372 | 0.500581 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| taker_long_short_volume_ratio_5m | bottom_20 | paper_short | low_fee_round_trip | 0.66130790 | 2.00 | -1.3387 | 0.480557 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| imbalance_1pct | bottom_20 | paper_short | low_fee_round_trip | 0.59571460 | 2.00 | -1.4043 | 0.502401 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| imbalance_5pct | top_20 | paper_long | low_fee_round_trip | 0.47073228 | 2.00 | -1.5293 | 0.493304 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| imbalance_5pct | bottom_20 | paper_short | low_fee_round_trip | 0.43621187 | 2.00 | -1.5638 | 0.492569 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| premium_index_1m | top_20 | paper_short | low_fee_round_trip | 0.37165082 | 2.00 | -1.6283 | 0.491429 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| open_interest_value_5m | bottom_20 | paper_short | low_fee_round_trip | 0.09872878 | 2.00 | -1.9013 | 0.459330 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| account_long_short_ratio_5m | top_20 | paper_short | low_fee_round_trip | 0.08075039 | 2.00 | -1.9192 | 0.470944 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| top_trader_long_short_ratio_5m | top_20 | paper_short | low_fee_round_trip | 0.06896910 | 2.00 | -1.9310 | 0.477685 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| open_interest_value_5m | top_20 | paper_short | low_fee_round_trip | 0.06276972 | 2.00 | -1.9372 | 0.500581 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| taker_long_short_volume_ratio_5m | bottom_20 | paper_short | taker_round_trip | 0.66130790 | 8.00 | -7.3387 | 0.480557 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| imbalance_1pct | bottom_20 | paper_short | taker_round_trip | 0.59571460 | 8.00 | -7.4043 | 0.502401 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| imbalance_5pct | top_20 | paper_long | taker_round_trip | 0.47073228 | 8.00 | -7.5293 | 0.493304 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| imbalance_5pct | bottom_20 | paper_short | taker_round_trip | 0.43621187 | 8.00 | -7.5638 | 0.492569 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| premium_index_1m | top_20 | paper_short | taker_round_trip | 0.37165082 | 8.00 | -7.6283 | 0.491429 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| open_interest_value_5m | bottom_20 | paper_short | taker_round_trip | 0.09872878 | 8.00 | -7.9013 | 0.459330 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| account_long_short_ratio_5m | top_20 | paper_short | taker_round_trip | 0.08075039 | 8.00 | -7.9192 | 0.470944 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| top_trader_long_short_ratio_5m | top_20 | paper_short | taker_round_trip | 0.06896910 | 8.00 | -7.9310 | 0.477685 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |
| open_interest_value_5m | top_20 | paper_short | taker_round_trip | 0.06276972 | 8.00 | -7.9372 | 0.500581 | not_viable_after_cost | reject for current execution mode; only revisit if execution cost or horizon changes |

## Summary

- maker_only_low_hit_rate: 1
- maker_or_internalized_candidate: 1
- not_viable_after_cost: 25
- zero_cost_only_signal: 9
- best viability: imbalance_1pct/bottom_20/paper_short/maker_or_internalized gross=0.59571460bps net=0.09571460bps status=maker_or_internalized_candidate
- best raw net diagnostic: taker_long_short_volume_ratio_5m/bottom_20/paper_short/zero_cost_diagnostic gross=0.66130790bps net=0.66130790bps
