# Current Action Preference OOS Check

This checks whether action preferences found in initial paper samples survive repeat samples. It is an OOS-shaped guardrail, not a final backtest or trained policy.

| candidate | context | asset | action | train n | train mean | test n | test mean | test hit | score | decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| near_microstructure_flow_paper_long | microstructure_flow | NEAR | paper_long | 2 | 148.70 | 3 | 81.23 | 0.667 | 130.10 | oos_supported_action_preference |
| volume_price_dislocation_paper_long | volume_price_dislocation |  | paper_long | 7 | -2.90 | 4 | 22.21 | 0.500 | 51.63 | oos_supported_action_preference |
| repeat_execution_paper_long | repeat_execution |  | paper_long | 2 | 56.82 | 4 | 10.40 | 0.500 | 48.08 | oos_supported_action_preference |
| sui_repeat_execution_paper_long | repeat_execution | SUI | paper_long | 2 | 56.82 | 4 | 10.40 | 0.500 | 48.08 | oos_supported_action_preference |
| sol_volume_price_dislocation_paper_long | volume_price_dislocation | SOL | paper_long | 1 | 73.78 | 3 | 30.30 | 0.667 | 67.99 | needs_repeat_oos |
| liquidation_intensity_paper_long | liquidation_intensity |  | paper_long | 0 | 0.00 | 2 | 12.41 | 0.500 | 29.27 | needs_repeat_oos |
| sui_liquidation_intensity_paper_long | liquidation_intensity | SUI | paper_long | 0 | 0.00 | 2 | 12.41 | 0.500 | 29.27 | needs_repeat_oos |
| sui_microstructure_flow_paper_long | microstructure_flow | SUI | paper_long | 0 | 0.00 | 2 | 8.40 | 0.500 | 26.60 | needs_repeat_oos |

## Interpretation

A passing row means the same context/action preference has repeat-sample support. A failing row means the apparent edge is likely first-sample overfit, timing luck, or missing execution/friction modeling.
