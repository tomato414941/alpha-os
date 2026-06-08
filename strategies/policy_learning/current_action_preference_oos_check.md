# Current Action Preference OOS Check

This checks whether action preferences found in initial paper samples survive repeat samples. It is an OOS-shaped guardrail, not a final backtest or trained policy.

| candidate | context | asset | action | train n | train mean | test n | test mean | test hit | score | decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| repeat_execution_paper_long | repeat_execution |  | paper_long | 3 | 186.49 | 5 | 99.64 | 0.800 | 168.62 | oos_supported_action_preference |
| sui_repeat_execution_paper_long | repeat_execution | SUI | paper_long | 2 | 106.50 | 5 | 99.64 | 0.800 | 151.29 | oos_supported_action_preference |
| sol_volume_price_dislocation_paper_long | volume_price_dislocation | SOL | paper_long | 2 | 104.37 | 4 | 76.30 | 0.750 | 126.23 | oos_supported_action_preference |
| eth_microstructure_flow_paper_long | microstructure_flow | ETH | paper_long | 2 | 98.37 | 2 | 48.43 | 0.500 | 68.12 | oos_supported_action_preference |
| sol_unclassified_paper_long | unclassified | SOL | paper_long | 4 | 83.79 | 2 | 31.61 | 0.500 | 59.64 | oos_supported_action_preference |
| liquidation_intensity_paper_long | liquidation_intensity |  | paper_long | 0 | 0.00 | 2 | 126.68 | 1.000 | 120.45 | needs_repeat_oos |
| sui_liquidation_intensity_paper_long | liquidation_intensity | SUI | paper_long | 0 | 0.00 | 2 | 126.68 | 1.000 | 120.45 | needs_repeat_oos |
| sui_microstructure_flow_paper_long | microstructure_flow | SUI | paper_long | 0 | 0.00 | 2 | 122.62 | 1.000 | 117.75 | needs_repeat_oos |
| stablecoin_migration_paper_long | stablecoin_migration |  | paper_long | 2 | 104.37 | 1 | 63.22 | 1.000 | 69.51 | needs_repeat_oos |
| sol_stablecoin_migration_paper_long | stablecoin_migration | SOL | paper_long | 2 | 104.37 | 1 | 63.22 | 1.000 | 69.51 | needs_repeat_oos |
| fartcoin_volume_price_dislocation_paper_long | volume_price_dislocation | FARTCOIN | paper_long | 1 | 188.47 | 1 | 0.00 | 0.000 | 17.42 | needs_repeat_oos |
| pump_volume_price_dislocation_paper_long | volume_price_dislocation | PUMP | paper_long | 1 | 162.81 | 1 | 0.00 | 0.000 | 16.14 | needs_repeat_oos |
| btc_unclassified_paper_long | unclassified | BTC | paper_long | 5 | 51.01 | 1 | 0.00 | 0.000 | 15.65 | needs_repeat_oos |
| xpl_volume_price_dislocation_paper_long | volume_price_dislocation | XPL | paper_long | 1 | 137.56 | 1 | 0.00 | 0.000 | 14.88 | needs_repeat_oos |
| event_pressure_paper_long | event_pressure |  | paper_long | 1 | 117.47 | 1 | 0.00 | 0.000 | 13.87 | needs_repeat_oos |
| aave_event_pressure_paper_long | event_pressure | AAVE | paper_long | 1 | 117.47 | 1 | 0.00 | 0.000 | 13.87 | needs_repeat_oos |
| link_volume_price_dislocation_paper_long | volume_price_dislocation | LINK | paper_long | 1 | 74.08 | 1 | 0.00 | 0.000 | 11.70 | needs_repeat_oos |
| sol_execution_edge_paper_long | execution_edge | SOL | paper_long | 2 | 63.22 | 0 | 0.00 | 0.000 | 11.32 | needs_repeat_oos |
| apt_unclassified_paper_long | unclassified | APT | paper_long | 1 | 44.51 | 1 | 0.00 | 0.000 | 10.23 | needs_repeat_oos |
| microstructure_flow_paper_long | microstructure_flow |  | paper_long | 7 | 233.43 | 15 | 38.07 | 0.400 | 100.66 | oos_failed_action_preference |
| hype_volume_price_dislocation_paper_long | volume_price_dislocation | HYPE | paper_long | 3 | 506.79 | 2 | 0.00 | 0.000 | 87.02 | oos_failed_action_preference |
| hype_microstructure_flow_paper_long | microstructure_flow | HYPE | paper_long | 3 | 499.56 | 2 | 0.00 | 0.000 | 85.93 | oos_failed_action_preference |
| volume_price_dislocation_paper_long | volume_price_dislocation |  | paper_long | 14 | 176.36 | 13 | 30.93 | 0.308 | 82.90 | oos_failed_action_preference |
| eth_volume_price_dislocation_paper_long | volume_price_dislocation | ETH | paper_long | 3 | 95.65 | 3 | 32.28 | 0.333 | 70.63 | oos_failed_action_preference |
| zec_unclassified_paper_long | unclassified | ZEC | paper_long | 6 | 157.88 | 2 | 0.00 | 0.000 | 34.68 | oos_failed_action_preference |
| eth_unclassified_paper_long | unclassified | ETH | paper_long | 5 | 90.19 | 2 | 0.00 | 0.000 | 24.53 | oos_failed_action_preference |
| unclassified_paper_long | unclassified |  | paper_long | 23 | 109.80 | 9 | -13.69 | 0.111 | 22.90 | oos_failed_action_preference |

## Interpretation

A passing row means the same context/action preference has repeat-sample support. A failing row means the apparent edge is likely first-sample overfit, timing luck, or missing execution/friction modeling.
