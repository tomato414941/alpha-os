# Current Split First Lane Repeat Queue

This queues lane-level paper work after mixed clusters are split. It is not a live trade instruction and does not collapse lanes back into a symbol-level action.

| priority | action | cluster | lane | side | status | next step |
| ---: | --- | --- | --- | --- | --- | --- |
| 363.5390 | open_lane_repeat_probe | hype_paper_long | hype_volume_price_dislocation | long | paper_repeat_cost_adjusted_probe | open one HYPE long lane-repeat for hype_volume_price_dislocation with explicit fill and stop notes |
| 349.5767 | open_lane_repeat_probe | sol_paper_long | sol_volume_price_dislocation | long | paper_repeat_cost_adjusted_probe | open one SOL long lane-repeat for sol_volume_price_dislocation with explicit fill and stop notes |
| 245.2831 | open_lane_label | sol_paper_long | sol_1971905_event_crypto_hedge | long | event_crypto_hedge_after_refresh_candidate | label SOL/sol_1971905_event_crypto_hedge alone before any repeat |
| 237.0390 | open_lane_label | hype_paper_long | hype_maker_or_low_fee_small_execution_edge | long | execution_low_fee_comparison_candidate | label HYPE/hype_maker_or_low_fee_small_execution_edge alone before any repeat |
| 235.0390 | open_lane_label | hype_paper_long | hype_taker_small_execution_edge | long | execution_taker_repeat_candidate | label HYPE/hype_taker_small_execution_edge alone before any repeat |
| 232.4924 | open_lane_label | hype_paper_long | hype_event_pressure_cluster | long | two_source_event_pressure | label HYPE/hype_event_pressure_cluster alone before any repeat |
| 222.8924 | open_lane_label | hype_paper_long | hype_attention_price_context | long | attention_price_lag_candidate | label HYPE/hype_attention_price_context alone before any repeat |
| 221.7829 | open_lane_label | sol_paper_long | solana_stablecoin_migration | long | paper_chain_stablecoin_inflow_watch | label SOL/solana_stablecoin_migration alone before any repeat |
| 217.6151 | open_lane_label | zec_paper_long | public_hypertracker_example_zec_wallet_flow_actionability | long | wallet_position_follow_candidate | label ZEC/public_hypertracker_example_zec_wallet_flow_actionability alone before any repeat |
| 216.5529 | open_lane_label | sol_paper_long | sol_attention_price_context | long | attention_price_lag_candidate | label SOL/sol_attention_price_context alone before any repeat |
| 214.2430 | open_lane_label | zec_paper_long | zec_privacy_sector_rotation_context | long | sector_perp_repeat_candidate | label ZEC/zec_privacy_sector_rotation_context alone before any repeat |
| 212.8241 | open_lane_label | zec_paper_long | zec_quantum_resistant_sector_rotation_context | long | sector_perp_repeat_candidate | label ZEC/zec_quantum_resistant_sector_rotation_context alone before any repeat |
| 212.0782 | open_lane_label | zec_paper_long | zec_zero_knowledge_zk_sector_rotation_context | long | sector_perp_repeat_candidate | label ZEC/zec_zero_knowledge_zk_sector_rotation_context alone before any repeat |
| 211.8869 | open_lane_label | zec_paper_long | zec_privacy_coins_sector_rotation_context | long | sector_perp_repeat_candidate | label ZEC/zec_privacy_coins_sector_rotation_context alone before any repeat |
| 208.4340 | open_lane_label | zec_paper_long | zec_event_pressure_cluster | long | two_source_event_pressure | label ZEC/zec_event_pressure_cluster alone before any repeat |
| 208.0651 | open_lane_label | sol_paper_long | sol_1962237_event_crypto_hedge | long | event_crypto_hedge_watch | label SOL/sol_1962237_event_crypto_hedge alone before any repeat |
| 198.6840 | open_lane_label | zec_paper_long | zec_attention_price_context | long | attention_price_lag_candidate | label ZEC/zec_attention_price_context alone before any repeat |
| 197.7703 | open_lane_label | hype_paper_long | hype_crowded_momentum_continuation_actionability | long | dislocation_repeat_needs_execution_check | label HYPE/hype_crowded_momentum_continuation_actionability alone before any repeat |
| 167.2525 | open_lane_label | zec_paper_long | zec_institutional_flow_news_event_quality_gate | long | repeat_single_source_label | label ZEC/zec_institutional_flow_news_event_quality_gate alone before any repeat |
| 167.2028 | open_lane_label | zec_paper_long | zec_narrative_event_news_event_quality_gate | long | repeat_single_source_label | label ZEC/zec_narrative_event_news_event_quality_gate alone before any repeat |
