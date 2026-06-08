# Current Action Preference OOS Check

This checks whether action preferences found in initial paper samples survive repeat samples. It is an OOS-shaped guardrail, not a final backtest or trained policy.

| candidate | context | asset | action | train n | train mean | test n | test mean | test hit | score | decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| repeat_execution_paper_long | repeat_execution |  | paper_long | 3 | 186.49 | 4 | 124.65 | 1.000 | 199.62 | oos_supported_action_preference |
| sui_repeat_execution_paper_long | repeat_execution | SUI | paper_long | 2 | 106.50 | 4 | 124.65 | 1.000 | 182.30 | oos_supported_action_preference |
| sol_volume_price_dislocation_paper_long | volume_price_dislocation | SOL | paper_long | 2 | 68.50 | 3 | 77.88 | 1.000 | 128.73 | oos_supported_action_preference |
| microstructure_flow_paper_long | microstructure_flow |  | paper_long | 7 | 123.99 | 12 | 69.17 | 0.583 | 120.84 | oos_supported_action_preference |
| volume_price_dislocation_paper_long | volume_price_dislocation |  | paper_long | 14 | 127.68 | 4 | 57.89 | 0.750 | 115.47 | oos_supported_action_preference |
| near_microstructure_flow_paper_long | microstructure_flow | NEAR | paper_long | 2 | 148.70 | 3 | 19.09 | 0.667 | 67.96 | oos_supported_action_preference |
| liquidation_intensity_paper_long | liquidation_intensity |  | paper_long | 0 | 0.00 | 2 | 126.68 | 1.000 | 120.45 | needs_repeat_oos |
| sui_liquidation_intensity_paper_long | liquidation_intensity | SUI | paper_long | 0 | 0.00 | 2 | 126.68 | 1.000 | 120.45 | needs_repeat_oos |
| sui_microstructure_flow_paper_long | microstructure_flow | SUI | paper_long | 0 | 0.00 | 2 | 122.62 | 1.000 | 117.75 | needs_repeat_oos |
| hype_volume_price_dislocation_paper_long | volume_price_dislocation | HYPE | paper_long | 3 | 335.81 | 0 | 0.00 | 0.000 | 53.70 | needs_repeat_oos |
| zec_unclassified_paper_long | unclassified | ZEC | paper_long | 5 | 96.64 | 0 | 0.00 | 0.000 | 17.50 | needs_repeat_oos |
| eth_volume_price_dislocation_paper_long | volume_price_dislocation | ETH | paper_long | 3 | 62.64 | 1 | -2.06 | 0.000 | 16.71 | needs_repeat_oos |
| eth_microstructure_flow_paper_long | microstructure_flow | ETH | paper_long | 2 | 48.87 | 1 | -2.06 | 0.000 | 12.20 | needs_repeat_oos |
| stablecoin_migration_paper_long | stablecoin_migration |  | paper_long | 2 | 68.50 | 0 | 0.00 | 0.000 | 11.85 | needs_repeat_oos |
| sol_stablecoin_migration_paper_long | stablecoin_migration | SOL | paper_long | 2 | 68.50 | 0 | 0.00 | 0.000 | 11.85 | needs_repeat_oos |
| sol_unclassified_paper_long | unclassified | SOL | paper_long | 4 | 34.25 | 0 | 0.00 | 0.000 | 7.64 | needs_repeat_oos |

## Interpretation

A passing row means the same context/action preference has repeat-sample support. A failing row means the apparent edge is likely first-sample overfit, timing luck, or missing execution/friction modeling.
