# Current LOB Policy Candidate Survival

This compares static LOB world replay with rolling sequence-state probes. A row is a policy-candidate diagnostic only when an action survives execution costs; zero-cost representation rows are kept separate.

| state family | action | mode | status | score | world net | seq net | zero-cost seq | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| taker_pressure | paper_short | maker_or_internalized_limit | lob_world_execution_with_sequence_representation | 105.6921 | 0.1613 | -0.2139 | 0.2861 | static replay survives execution, and rolling sequence keeps representation value |
| order_book_imbalance | paper_short | maker_or_internalized_limit | lob_world_execution_with_sequence_representation | 104.3637 | 0.0957 | -0.0986 | 0.4014 | static replay survives execution, and rolling sequence keeps representation value |
| liquidity_change | paper_short | maker_or_internalized_limit | lob_sequence_execution_probe | 79.1793 | 0.0000 | 0.0475 | 0.5475 | rolling sequence state survives execution costs, but static replay does not confirm it |
| liquidity_change | paper_short | zero_cost_representation | lob_representation_only | 56.6793 | 0.0000 | 0.5475 | 0.5475 | state representation has signal before execution costs only |
| basis_premium | paper_long | zero_cost_representation | lob_representation_only | 47.8631 | 0.0000 | 0.4159 | 0.4159 | state representation has signal before execution costs only |
| order_book_imbalance | paper_short | zero_cost_representation | lob_representation_only | 46.8919 | 0.0000 | 0.4014 | 0.4014 | state representation has signal before execution costs only |
| taker_pressure | paper_short | zero_cost_representation | lob_representation_only | 39.1691 | 0.0000 | 0.2861 | 0.2861 | state representation has signal before execution costs only |
| basis_premium | paper_short | zero_cost_representation | lob_representation_only | 37.5983 | 0.0000 | 0.2627 | 0.2627 | state representation has signal before execution costs only |
| order_book_imbalance | paper_long | zero_cost_representation | lob_representation_only | 36.3453 | 0.0000 | 0.2440 | 0.2440 | state representation has signal before execution costs only |
| liquidity_change | paper_short | low_fee_cross | lob_policy_rejected_after_cost | -43.4306 | 0.0000 | -1.4525 | 0.5475 | state/action does not survive the current execution-cost check |
| liquidity_change | paper_short | market_order | lob_policy_rejected_after_cost | -43.4306 | 0.0000 | -7.4525 | 0.5475 | state/action does not survive the current execution-cost check |
| order_book_imbalance | paper_short | low_fee_cross | lob_policy_rejected_after_cost | -44.9434 | -1.4043 | -1.5986 | 0.4014 | state/action does not survive the current execution-cost check |
| order_book_imbalance | paper_short | market_order | lob_policy_rejected_after_cost | -44.9434 | -7.4043 | -7.5986 | 0.4014 | state/action does not survive the current execution-cost check |
| basis_premium | paper_long | low_fee_cross | lob_policy_rejected_after_cost | -45.0096 | -2.1302 | -1.5841 | 0.4159 | state/action does not survive the current execution-cost check |
| basis_premium | paper_long | maker_or_internalized_limit | lob_policy_rejected_after_cost | -45.0096 | -0.6302 | -0.0841 | 0.4159 | state/action does not survive the current execution-cost check |
| basis_premium | paper_long | market_order | lob_policy_rejected_after_cost | -45.0096 | -8.1302 | -7.5841 | 0.4159 | state/action does not survive the current execution-cost check |
| taker_pressure | paper_short | low_fee_cross | lob_policy_rejected_after_cost | -46.5667 | -1.3387 | -1.7139 | 0.2861 | state/action does not survive the current execution-cost check |
| taker_pressure | paper_short | market_order | lob_policy_rejected_after_cost | -46.5667 | -7.3387 | -7.7139 | 0.2861 | state/action does not survive the current execution-cost check |
| basis_premium | paper_short | low_fee_cross | lob_policy_rejected_after_cost | -46.8481 | -1.6283 | -1.7373 | 0.2627 | state/action does not survive the current execution-cost check |
| basis_premium | paper_short | maker_or_internalized_limit | lob_policy_rejected_after_cost | -46.8481 | -0.1283 | -0.2373 | 0.2627 | state/action does not survive the current execution-cost check |
| basis_premium | paper_short | market_order | lob_policy_rejected_after_cost | -46.8481 | -7.6283 | -7.7373 | 0.2627 | state/action does not survive the current execution-cost check |
| order_book_imbalance | paper_long | low_fee_cross | lob_policy_rejected_after_cost | -47.0725 | -1.5293 | -1.7560 | 0.2440 | state/action does not survive the current execution-cost check |
| order_book_imbalance | paper_long | maker_or_internalized_limit | lob_policy_rejected_after_cost | -47.0725 | -0.0293 | -0.2560 | 0.2440 | state/action does not survive the current execution-cost check |
| order_book_imbalance | paper_long | market_order | lob_policy_rejected_after_cost | -47.0725 | -7.5293 | -7.7560 | 0.2440 | state/action does not survive the current execution-cost check |
| liquidity_change | paper_long | low_fee_cross | lob_policy_rejected_after_cost | -50.0000 | 0.0000 | -2.3623 | -0.3623 | state/action does not survive the current execution-cost check |
| liquidity_change | paper_long | maker_or_internalized_limit | lob_policy_rejected_after_cost | -50.0000 | 0.0000 | -0.8623 | -0.3623 | state/action does not survive the current execution-cost check |
| liquidity_change | paper_long | market_order | lob_policy_rejected_after_cost | -50.0000 | 0.0000 | -8.3623 | -0.3623 | state/action does not survive the current execution-cost check |
| liquidity_change | paper_long | zero_cost_representation | lob_policy_rejected_after_cost | -50.0000 | 0.0000 | -0.3623 | -0.3623 | state/action does not survive the current execution-cost check |
| positioning | paper_long | low_fee_cross | lob_policy_rejected_after_cost | -50.0000 | -2.1701 | -3.0414 | -1.0414 | state/action does not survive the current execution-cost check |
| positioning | paper_long | maker_or_internalized_limit | lob_policy_rejected_after_cost | -50.0000 | -0.6701 | -1.5414 | -1.0414 | state/action does not survive the current execution-cost check |
| positioning | paper_long | market_order | lob_policy_rejected_after_cost | -50.0000 | -8.1701 | -9.0414 | -1.0414 | state/action does not survive the current execution-cost check |
| positioning | paper_long | zero_cost_representation | lob_policy_rejected_after_cost | -50.0000 | 0.0000 | -1.0414 | -1.0414 | state/action does not survive the current execution-cost check |
| positioning | paper_short | low_fee_cross | lob_policy_rejected_after_cost | -50.0000 | -1.9013 | -2.4213 | -0.4213 | state/action does not survive the current execution-cost check |
| positioning | paper_short | maker_or_internalized_limit | lob_policy_rejected_after_cost | -50.0000 | -0.4013 | -0.9213 | -0.4213 | state/action does not survive the current execution-cost check |
| positioning | paper_short | market_order | lob_policy_rejected_after_cost | -50.0000 | -7.9013 | -8.4213 | -0.4213 | state/action does not survive the current execution-cost check |
| positioning | paper_short | zero_cost_representation | lob_policy_rejected_after_cost | -50.0000 | 0.0000 | -0.4213 | -0.4213 | state/action does not survive the current execution-cost check |
