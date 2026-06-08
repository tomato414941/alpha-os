# Current Action Preference Candidates

This aggregates RL-shaped paper samples into context/action preferences. It is not a trained policy and not a deployable strategy.

| candidate | scope | context | asset | action | samples | hit | mean | median | worst | score | decision |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| near_microstructure_flow_paper_long | asset_family_action | microstructure_flow | NEAR | paper_long | 5 | 0.800 | 108.22 | 133.13 | 0.00 | 160.72 | promote_action_preference_candidate |
| sol_volume_price_dislocation_paper_long | asset_family_action | volume_price_dislocation | SOL | paper_long | 4 | 0.750 | 41.17 | 46.93 | -2.95 | 89.92 | promote_action_preference_candidate |
| repeat_execution_paper_long | family_action | repeat_execution |  | paper_long | 6 | 0.667 | 25.87 | 21.87 | -1.06 | 77.54 | promote_action_preference_candidate |
| sui_repeat_execution_paper_long | asset_family_action | repeat_execution | SUI | paper_long | 6 | 0.667 | 25.87 | 21.87 | -1.06 | 77.54 | promote_action_preference_candidate |
| volume_price_dislocation_paper_long | family_action | volume_price_dislocation |  | paper_long | 9 | 0.556 | 6.04 | 7.55 | -248.86 | 42.48 | watch_action_preference_candidate |
| liquidation_intensity_paper_long | family_action | liquidation_intensity |  | paper_long | 2 | 0.500 | 12.41 | 12.41 | -1.06 | 28.21 | watch_action_preference_candidate |
| sui_liquidation_intensity_paper_long | asset_family_action | liquidation_intensity | SUI | paper_long | 2 | 0.500 | 12.41 | 12.41 | -1.06 | 28.21 | watch_action_preference_candidate |
| sui_microstructure_flow_paper_long | asset_family_action | microstructure_flow | SUI | paper_long | 2 | 0.500 | 8.40 | 8.40 | -1.06 | 22.86 | watch_action_preference_candidate |
| inj_volume_price_dislocation_paper_long | asset_family_action | volume_price_dislocation | INJ | paper_long | 1 | 1.000 | 153.68 | 153.68 | 153.68 | 67.06 | collect_more_labels |
| microstructure_flow_paper_long | family_action | microstructure_flow |  | paper_long | 17 | 0.353 | 2.33 | 0.00 | -266.52 | 26.15 | collect_more_labels |
| eth_volume_price_dislocation_paper_long | asset_family_action | volume_price_dislocation | ETH | paper_long | 2 | 0.500 | 2.74 | 2.74 | -2.06 | 15.32 | collect_more_labels |
| eth_microstructure_flow_paper_long | asset_family_action | microstructure_flow | ETH | paper_long | 2 | 0.500 | 2.74 | 2.74 | -2.06 | 15.32 | collect_more_labels |
| unclassified_paper_long | family_action | unclassified |  | paper_long | 12 | 0.000 | 0.00 | 0.00 | 0.00 | 15.00 | collect_more_labels |
| btc_unclassified_paper_long | asset_family_action | unclassified | BTC | paper_long | 3 | 0.000 | 0.00 | 0.00 | 0.00 | 7.50 | collect_more_labels |
| zec_unclassified_paper_long | asset_family_action | unclassified | ZEC | paper_long | 3 | 0.000 | 0.00 | 0.00 | 0.00 | 7.50 | collect_more_labels |
| eth_unclassified_paper_long | asset_family_action | unclassified | ETH | paper_long | 2 | 0.000 | 0.00 | 0.00 | 0.00 | 3.33 | collect_more_labels |
| event_unclassified_paper_long | asset_family_action | unclassified | EVENT | paper_long | 2 | 0.000 | 0.00 | 0.00 | 0.00 | 3.33 | collect_more_labels |
| execution_edge_paper_long | family_action | execution_edge |  | paper_long | 2 | 0.000 | 0.00 | 0.00 | 0.00 | 3.33 | collect_more_labels |
| near_execution_edge_paper_long | asset_family_action | execution_edge | NEAR | paper_long | 2 | 0.000 | 0.00 | 0.00 | 0.00 | 3.33 | collect_more_labels |
| sol_unclassified_paper_long | asset_family_action | unclassified | SOL | paper_long | 1 | 0.000 | 0.00 | 0.00 | 0.00 | 0.83 | collect_more_labels |
| bera_microstructure_flow_paper_long | asset_family_action | microstructure_flow | BERA | paper_long | 1 | 0.000 | 0.00 | 0.00 | 0.00 | 0.83 | collect_more_labels |
| sei_microstructure_flow_paper_long | asset_family_action | microstructure_flow | SEI | paper_long | 1 | 0.000 | 0.00 | 0.00 | 0.00 | 0.83 | collect_more_labels |
| near_unclassified_paper_long | asset_family_action | unclassified | NEAR | paper_long | 1 | 0.000 | 0.00 | 0.00 | 0.00 | 0.83 | collect_more_labels |
| chip_microstructure_flow_paper_long | asset_family_action | microstructure_flow | CHIP | paper_long | 1 | 0.000 | 0.00 | 0.00 | 0.00 | 0.83 | collect_more_labels |
| protocol_fee_paper_short | family_action | protocol_fee |  | paper_short | 2 | 0.500 | -12.81 | -12.81 | -37.55 | -5.42 | collect_more_labels |
| hype_protocol_fee_paper_short | asset_family_action | protocol_fee | HYPE | paper_short | 2 | 0.500 | -12.81 | -12.81 | -37.55 | -5.42 | collect_more_labels |
| mon_microstructure_flow_paper_long | asset_family_action | microstructure_flow | MON | paper_long | 1 | 0.000 | -32.04 | -32.04 | -32.04 | -20.52 | collect_more_labels |
| protocol_fee_paper_long | family_action | protocol_fee |  | paper_long | 1 | 0.000 | -66.97 | -66.97 | -66.97 | -43.81 | collect_more_labels |
| crv_protocol_fee_paper_long | asset_family_action | protocol_fee | CRV | paper_long | 1 | 0.000 | -66.97 | -66.97 | -66.97 | -43.81 | collect_more_labels |
| token_unlock_paper_short | family_action | token_unlock |  | paper_short | 4 | 0.250 | -9.39 | -5.98 | -37.55 | 0.88 | reject_action_preference_candidate |
| hype_token_unlock_paper_short | asset_family_action | token_unlock | HYPE | paper_short | 4 | 0.250 | -9.39 | -5.98 | -37.55 | 0.88 | reject_action_preference_candidate |
| hype_microstructure_flow_paper_long | asset_family_action | microstructure_flow | HYPE | paper_long | 2 | 0.000 | -20.62 | -20.62 | -20.62 | -24.15 | reject_action_preference_candidate |
| unclassified_paper_short | family_action | unclassified |  | paper_short | 2 | 0.000 | -104.14 | -104.14 | -126.57 | -135.53 | reject_action_preference_candidate |
| sol_unclassified_paper_short | asset_family_action | unclassified | SOL | paper_short | 2 | 0.000 | -104.14 | -104.14 | -126.57 | -135.53 | reject_action_preference_candidate |
| hype_volume_price_dislocation_paper_long | asset_family_action | volume_price_dislocation | HYPE | paper_long | 2 | 0.000 | -134.74 | -134.74 | -248.86 | -176.32 | reject_action_preference_candidate |
| arb_intraday_derivatives_paper_short | asset_family_action | intraday_derivatives | ARB | paper_short | 14 | 0.000 | -122.24 | -89.22 | -166.26 | -196.46 | reject_action_preference_candidate |
| intraday_derivatives_paper_short | family_action | intraday_derivatives |  | paper_short | 19 | 0.000 | -139.41 | -158.66 | -302.94 | -283.07 | reject_action_preference_candidate |
| mega_microstructure_flow_paper_long | asset_family_action | microstructure_flow | MEGA | paper_long | 2 | 0.000 | -225.23 | -225.23 | -266.52 | -296.98 | reject_action_preference_candidate |
| near_intraday_derivatives_paper_short | asset_family_action | intraday_derivatives | NEAR | paper_short | 5 | 0.000 | -187.51 | -158.66 | -302.94 | -333.67 | reject_action_preference_candidate |

## Interpretation

A candidate means the current paper logs suggest an action preference for a context. It still needs a leakage-safe split, more samples, explicit cost/fill assumptions, and a clear policy evaluation protocol before it can influence live decisions.
