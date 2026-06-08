# Current LOB Maker Fill Survival

This is a conservative maker-fill proxy over existing book-depth samples. A passive order is considered filled only when the next 1m mark crosses the passive offset; the reward is the filled mark reward after maker cost. It is a fill/adverse-selection gate, not a live execution model.

| candidate | source | status | score | test n | fills | fill rate | filled bps | all-state bps | adverse | optimistic net | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| taker_pressure_world_replay_taker_long_short_volume_ratio_5m_bottom_20_paper_short | world_replay | maker_adverse_selection_blocked | -361.2718 | 13218 | 6176 | 0.467242 | -11.4559 | -5.3527 | 1.000000 | 0.1613 | passive fills are mostly adverse after the mark crosses the order |
| order_book_imbalance_world_replay_imbalance_1pct_bottom_20_paper_short | world_replay | maker_adverse_selection_blocked | -375.3766 | 21660 | 9767 | 0.450923 | -12.4051 | -5.5937 | 1.000000 | 0.0957 | passive fills are mostly adverse after the mark crosses the order |
| taker_pressure_sequence_state_taker_pressure_delta_5_bottom_20_paper_short | sequence_state | maker_adverse_selection_blocked | -378.9242 | 15920 | 7446 | 0.467714 | -12.2067 | -5.7092 | 1.000000 | -0.2139 | passive fills are mostly adverse after the mark crosses the order |
| liquidity_change_sequence_state_ask_liquidity_change_5_top_20_paper_short | sequence_state | maker_adverse_selection_blocked | -403.5075 | 21341 | 9694 | 0.454243 | -13.8060 | -6.2713 | 1.000000 | 0.0475 | passive fills are mostly adverse after the mark crosses the order |
| order_book_imbalance_sequence_state_shallow_deep_imbalance_gap_bottom_20_paper_short | sequence_state | maker_adverse_selection_blocked | -409.9347 | 19180 | 8605 | 0.448644 | -14.2132 | -6.3767 | 1.000000 | -0.0986 | passive fills are mostly adverse after the mark crosses the order |
