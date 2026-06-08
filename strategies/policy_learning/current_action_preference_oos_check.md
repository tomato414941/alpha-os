# Current Action Preference OOS Check

This checks whether action preferences found in initial paper samples survive repeat samples. It is an OOS-shaped guardrail, not a final backtest or trained policy.

| candidate | context | asset | action | train n | train mean | test n | test mean | test hit | score | decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| repeat_execution_paper_long | repeat_execution |  | paper_long | 3 | 270.21 | 5 | 212.46 | 1.000 | 299.99 | oos_supported_action_preference |
| sui_repeat_execution_paper_long | repeat_execution | SUI | paper_long | 2 | 227.41 | 5 | 212.46 | 1.000 | 282.20 | oos_supported_action_preference |
| zec_news_event_paper_long | news_event | ZEC | paper_long | 4 | 346.42 | 4 | 175.30 | 1.000 | 274.26 | oos_supported_action_preference |
| sol_volume_price_dislocation_paper_long | volume_price_dislocation | SOL | paper_long | 3 | 183.69 | 4 | 191.16 | 1.000 | 265.71 | oos_supported_action_preference |
| news_event_paper_long | news_event |  | paper_long | 6 | 360.93 | 6 | 149.34 | 1.000 | 250.48 | oos_supported_action_preference |
| volume_price_dislocation_paper_long | volume_price_dislocation |  | paper_long | 22 | 204.45 | 14 | 107.84 | 1.000 | 185.51 | oos_supported_action_preference |
| sol_event_crypto_hedge_paper_long | event_crypto_hedge | SOL | paper_long | 3 | 179.96 | 4 | 107.88 | 1.000 | 181.88 | oos_supported_action_preference |
| microstructure_flow_paper_long | microstructure_flow |  | paper_long | 24 | 155.82 | 20 | 115.44 | 0.800 | 179.81 | oos_supported_action_preference |
| wallet_entity_flow_paper_long | wallet_entity_flow |  | paper_long | 3 | 210.48 | 4 | 91.48 | 1.000 | 170.05 | oos_supported_action_preference |
| eth_volume_price_dislocation_paper_long | volume_price_dislocation | ETH | paper_long | 5 | 124.69 | 3 | 96.92 | 1.000 | 159.62 | oos_supported_action_preference |
| stablecoin_migration_paper_long | stablecoin_migration |  | paper_long | 2 | 221.59 | 2 | 143.92 | 1.000 | 159.11 | oos_supported_action_preference |
| sol_stablecoin_migration_paper_long | stablecoin_migration | SOL | paper_long | 2 | 221.59 | 2 | 143.92 | 1.000 | 159.11 | oos_supported_action_preference |
| near_microstructure_flow_paper_long | microstructure_flow | NEAR | paper_long | 9 | 171.19 | 3 | 79.69 | 0.667 | 139.37 | oos_supported_action_preference |
| eth_microstructure_flow_paper_long | microstructure_flow | ETH | paper_long | 5 | 104.66 | 2 | 114.73 | 1.000 | 133.19 | oos_supported_action_preference |
| eth_event_crypto_hedge_paper_long | event_crypto_hedge | ETH | paper_long | 4 | 161.46 | 5 | 61.29 | 1.000 | 132.51 | oos_supported_action_preference |
| event_crypto_hedge_paper_long | event_crypto_hedge |  | paper_long | 11 | 140.04 | 14 | 63.14 | 1.000 | 131.14 | oos_supported_action_preference |
| sol_execution_edge_paper_long | execution_edge | SOL | paper_long | 2 | 179.96 | 2 | 107.88 | 1.000 | 130.92 | oos_supported_action_preference |
| hype_volume_price_dislocation_paper_long | volume_price_dislocation | HYPE | paper_long | 5 | 324.63 | 3 | 14.87 | 1.000 | 107.56 | oos_supported_action_preference |
| btc_event_crypto_hedge_paper_long | event_crypto_hedge | BTC | paper_long | 4 | 88.69 | 5 | 29.18 | 1.000 | 89.49 | oos_supported_action_preference |
| hype_microstructure_flow_paper_long | microstructure_flow | HYPE | paper_long | 10 | 167.58 | 4 | 14.87 | 1.000 | 87.00 | oos_supported_action_preference |
| bera_microstructure_flow_paper_long | microstructure_flow | BERA | paper_long | 0 | 0.00 | 2 | 435.24 | 1.000 | 326.16 | needs_repeat_oos |
| liquidation_intensity_paper_long | liquidation_intensity |  | paper_long | 0 | 0.00 | 2 | 239.79 | 1.000 | 195.86 | needs_repeat_oos |
| sui_liquidation_intensity_paper_long | liquidation_intensity | SUI | paper_long | 0 | 0.00 | 2 | 239.79 | 1.000 | 195.86 | needs_repeat_oos |
| sui_microstructure_flow_paper_long | microstructure_flow | SUI | paper_long | 0 | 0.00 | 2 | 235.69 | 1.000 | 193.13 | needs_repeat_oos |
| chip_microstructure_flow_paper_long | microstructure_flow | CHIP | paper_long | 0 | 0.00 | 2 | 174.18 | 1.000 | 152.12 | needs_repeat_oos |
| zec_wallet_entity_flow_paper_long | wallet_entity_flow | ZEC | paper_long | 1 | 346.42 | 1 | 175.30 | 1.000 | 113.75 | needs_repeat_oos |
| sector_rotation_paper_long | sector_rotation |  | paper_long | 1 | 346.42 | 1 | 175.30 | 1.000 | 113.75 | needs_repeat_oos |
| zec_sector_rotation_paper_long | sector_rotation | ZEC | paper_long | 1 | 346.42 | 1 | 175.30 | 1.000 | 113.75 | needs_repeat_oos |
| sol_news_event_paper_long | news_event | SOL | paper_long | 1 | 263.22 | 1 | 179.96 | 1.000 | 111.15 | needs_repeat_oos |
| link_volume_price_dislocation_paper_long | volume_price_dislocation | LINK | paper_long | 1 | 215.45 | 1 | 129.81 | 1.000 | 92.04 | needs_repeat_oos |
| fartcoin_volume_price_dislocation_paper_long | volume_price_dislocation | FARTCOIN | paper_long | 1 | 317.53 | 1 | 113.30 | 1.000 | 91.64 | needs_repeat_oos |
| xpl_volume_price_dislocation_paper_long | volume_price_dislocation | XPL | paper_long | 1 | 270.89 | 1 | 118.78 | 1.000 | 91.14 | needs_repeat_oos |
| eth_wallet_entity_flow_paper_long | wallet_entity_flow | ETH | paper_long | 1 | 161.46 | 2 | 61.29 | 1.000 | 89.94 | needs_repeat_oos |
| event_pressure_paper_long | event_pressure |  | paper_long | 1 | 228.43 | 1 | 97.91 | 1.000 | 82.06 | needs_repeat_oos |
| aave_event_pressure_paper_long | event_pressure | AAVE | paper_long | 1 | 228.43 | 1 | 97.91 | 1.000 | 82.06 | needs_repeat_oos |
| mon_microstructure_flow_paper_long | microstructure_flow | MON | paper_long | 0 | 0.00 | 2 | 83.55 | 0.500 | 76.70 | needs_repeat_oos |
| hype_news_event_paper_long | news_event | HYPE | paper_long | 1 | 516.65 | 1 | 14.87 | 1.000 | 68.79 | needs_repeat_oos |
| apt_wallet_entity_flow_paper_long | wallet_entity_flow | APT | paper_long | 1 | 123.57 | 1 | 68.01 | 1.000 | 66.85 | needs_repeat_oos |
| pump_volume_price_dislocation_paper_long | volume_price_dislocation | PUMP | paper_long | 1 | 226.23 | 1 | 47.92 | 1.000 | 65.28 | needs_repeat_oos |
| options_volatility_paper_long | options_volatility |  | paper_long | 1 | 88.69 | 1 | 29.18 | 1.000 | 52.16 | needs_repeat_oos |

## Interpretation

A passing row means the same context/action preference has repeat-sample support. A failing row means the apparent edge is likely first-sample overfit, timing luck, or missing execution/friction modeling.
