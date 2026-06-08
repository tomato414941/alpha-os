# Current Action Preference OOS Check

This checks whether action preferences found in initial paper samples survive repeat samples. It is an OOS-shaped guardrail, not a final backtest or trained policy.

| candidate | context | asset | action | train n | train mean | test n | test mean | test hit | score | decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| zec_news_event_paper_long | news_event | ZEC | paper_long | 7 | 491.79 | 6 | 461.89 | 1.000 | 582.66 | oos_supported_action_preference |
| zec_wallet_entity_flow_paper_long | wallet_entity_flow | ZEC | paper_long | 3 | 456.55 | 3 | 397.55 | 1.000 | 510.03 | oos_supported_action_preference |
| news_event_paper_long | news_event |  | paper_long | 18 | 176.54 | 8 | 345.53 | 0.875 | 412.76 | oos_supported_action_preference |
| zec_sector_rotation_paper_long | sector_rotation | ZEC | paper_long | 8 | 333.21 | 2 | 429.72 | 1.000 | 377.46 | oos_supported_action_preference |
| sector_rotation_paper_long | sector_rotation |  | paper_long | 9 | 292.51 | 2 | 429.72 | 1.000 | 370.80 | oos_supported_action_preference |
| wallet_entity_flow_paper_long | wallet_entity_flow |  | paper_long | 7 | 196.06 | 11 | 104.71 | 0.727 | 171.51 | oos_supported_action_preference |
| sol_volume_price_dislocation_paper_long | volume_price_dislocation | SOL | paper_long | 2 | 127.75 | 6 | 123.60 | 0.667 | 170.87 | oos_supported_action_preference |
| repeat_execution_paper_long | repeat_execution |  | paper_long | 3 | 283.91 | 6 | 53.89 | 0.667 | 133.48 | oos_supported_action_preference |
| near_microstructure_flow_paper_long | microstructure_flow | NEAR | paper_long | 2 | 198.04 | 10 | 65.74 | 1.000 | 132.54 | oos_supported_action_preference |
| execution_edge_paper_long | execution_edge |  | paper_long | 8 | 134.39 | 2 | 104.08 | 1.000 | 129.92 | oos_supported_action_preference |
| sol_execution_edge_paper_long | execution_edge | SOL | paper_long | 2 | 176.13 | 2 | 104.08 | 1.000 | 128.00 | oos_supported_action_preference |
| stablecoin_migration_paper_long | stablecoin_migration |  | paper_long | 2 | 127.75 | 4 | 68.13 | 0.500 | 110.40 | oos_supported_action_preference |
| sol_stablecoin_migration_paper_long | stablecoin_migration | SOL | paper_long | 2 | 127.75 | 4 | 68.13 | 0.500 | 110.40 | oos_supported_action_preference |
| sol_event_crypto_hedge_paper_long | event_crypto_hedge | SOL | paper_long | 4 | 86.14 | 7 | 57.82 | 0.571 | 102.39 | oos_supported_action_preference |
| bera_microstructure_flow_paper_long | microstructure_flow | BERA | paper_long | 0 | 0.00 | 2 | 517.91 | 1.000 | 381.28 | needs_repeat_oos |
| chip_microstructure_flow_paper_long | microstructure_flow | CHIP | paper_long | 0 | 0.00 | 2 | 477.95 | 1.000 | 354.63 | needs_repeat_oos |
| liquidation_intensity_paper_long | liquidation_intensity |  | paper_long | 0 | 0.00 | 2 | 113.78 | 1.000 | 111.85 | needs_repeat_oos |
| sui_liquidation_intensity_paper_long | liquidation_intensity | SUI | paper_long | 0 | 0.00 | 2 | 113.78 | 1.000 | 111.85 | needs_repeat_oos |
| sui_microstructure_flow_paper_long | microstructure_flow | SUI | paper_long | 0 | 0.00 | 2 | 109.73 | 1.000 | 109.15 | needs_repeat_oos |
| inj_volume_price_dislocation_paper_long | volume_price_dislocation | INJ | paper_long | 0 | 0.00 | 5 | 63.88 | 1.000 | 105.88 | needs_repeat_oos |
| apt_wallet_entity_flow_paper_long | wallet_entity_flow | APT | paper_long | 1 | 17.05 | 3 | 42.81 | 1.000 | 87.66 | needs_repeat_oos |
| sui_repeat_execution_paper_long | repeat_execution | SUI | paper_long | 0 | 0.00 | 6 | 53.89 | 0.667 | 85.89 | needs_repeat_oos |
| near_unclassified_paper_long | unclassified | NEAR | paper_long | 2 | 39.52 | 1 | 38.60 | 1.000 | 54.82 | needs_repeat_oos |
| chip_repeat_execution_paper_long | repeat_execution | CHIP | paper_long | 3 | 283.91 | 0 | 0.00 | 0.000 | 47.59 | needs_repeat_oos |
| chip_execution_edge_paper_long | execution_edge | CHIP | paper_long | 3 | 283.91 | 0 | 0.00 | 0.000 | 47.59 | needs_repeat_oos |
| zec_event_pressure_paper_long | event_pressure | ZEC | paper_long | 2 | 333.21 | 0 | 0.00 | 0.000 | 38.32 | needs_repeat_oos |
| protocol_fee_paper_long | protocol_fee |  | paper_long | 3 | 75.96 | 0 | 0.00 | 0.000 | 16.39 | needs_repeat_oos |
| uni_protocol_fee_paper_long | protocol_fee | UNI | paper_long | 3 | 75.96 | 0 | 0.00 | 0.000 | 16.39 | needs_repeat_oos |
| basis_term_structure_paper_short | basis_term_structure |  | paper_short | 2 | 63.25 | 0 | 0.00 | 0.000 | 11.32 | needs_repeat_oos |
| near_execution_edge_paper_long | execution_edge | NEAR | paper_long | 2 | 38.60 | 0 | 0.00 | 0.000 | 8.86 | needs_repeat_oos |
| microstructure_flow_paper_long | microstructure_flow |  | paper_long | 5 | 222.96 | 40 | -10.61 | 0.450 | 53.33 | oos_failed_action_preference |
| eth_event_crypto_hedge_paper_long | event_crypto_hedge | ETH | paper_long | 4 | 62.23 | 8 | -20.10 | 0.625 | 23.73 | oos_failed_action_preference |

## Interpretation

A passing row means the same context/action preference has repeat-sample support. A failing row means the apparent edge is likely first-sample overfit, timing luck, or missing execution/friction modeling.
