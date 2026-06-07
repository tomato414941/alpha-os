# Current Follow-Up Queue

This queue turns current labels into repeatable next observations. It is not a trading instruction; it is a work queue for finding which alpha source is real.

| priority | asset | source | type | action | evidence | next test |
| ---: | --- | --- | --- | --- | --- | --- |
| 10.0571 | WLD | hl_candidate;okx_pressure;liquidation | clean_candidate_repeat | repeat_supported_candidate | hl15=0.0197;okx_pressure15=0.0247;liq_cont15=0.0273 | repeat the same labels on fresh samples and add rough costs |
| 4.5510 | ETH | okx_pressure;liquidation;l2_imbalance | clean_candidate_repeat | repeat_supported_candidate | okx_pressure15=0.0007;liq_cont15=0.0011;l2_imbalance15=0.0010 | repeat the same labels on fresh samples and add rough costs |
| 3.8916 | MEGA | hl_candidate | clean_candidate_repeat | repeat_supported_candidate | hl15=0.0178 | repeat the same labels on fresh samples and add rough costs |
| 3.7269 | PEPE | okx_pressure;liquidation | clean_candidate_repeat | repeat_supported_candidate | okx_pressure15=0.0040;liq_cont15=0.0033 | repeat the same labels on fresh samples and add rough costs |
| 3.6217 | BTC | liquidation;l2_imbalance | source_isolation | repeat_liquidation_not_pressure | positive=liquidation;l2_imbalance;negative=okx_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| 3.6106 | ONDO | liquidation;sector_rotation | source_isolation | separate_sector_from_l2 | positive=liquidation;sector_rotation;negative=okx_pressure;l2_imbalance | repeat sector labels with category membership and costs before mixing with other sources |
| 3.6004 | XMR | hl_candidate | source_isolation | separate_carry_from_sector | positive=hl_candidate;negative=sector_rotation | repeat the original candidate family and keep unrelated negative sources out of the decision |
| 3.4627 | XRP | okx_pressure;liquidation | clean_candidate_repeat | repeat_supported_candidate | okx_pressure15=0.0026;liq_cont15=0.0018 | repeat the same labels on fresh samples and add rough costs |
| 3.4579 | JTO | liquidation;l2_imbalance | source_isolation | repeat_liquidation_not_pressure | positive=liquidation;l2_imbalance;negative=okx_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| 3.4493 | XPL | l2_imbalance;sector_rotation | source_isolation | repeat_l2_not_pressure | positive=l2_imbalance;sector_rotation;negative=okx_pressure | repeat sector labels with category membership and costs before mixing with other sources |
| 3.3166 | IP | hl_candidate | source_isolation | isolate_positive_source | positive=hl_candidate;negative=okx_pressure | repeat the original candidate family and keep unrelated negative sources out of the decision |
| 3.2959 | LTC | okx_pressure;liquidation | clean_candidate_repeat | repeat_supported_candidate | okx_pressure15=0.0002;liq_cont15=0.0026 | repeat the same labels on fresh samples and add rough costs |
| 3.2743 | ZORA | hl_candidate | clean_candidate_repeat | repeat_supported_candidate | hl15=0.0055 | repeat the same labels on fresh samples and add rough costs |
| 3.1882 | KAITO | hl_candidate | clean_candidate_repeat | repeat_supported_candidate | hl15=0.0038 | repeat the same labels on fresh samples and add rough costs |
| 3.1603 | AIXBT | hl_candidate | clean_candidate_repeat | repeat_supported_candidate | hl15=0.0032 | repeat the same labels on fresh samples and add rough costs |
| 3.1524 | APEX | hl_candidate | clean_candidate_repeat | repeat_supported_candidate | hl15=0.0030 | repeat the same labels on fresh samples and add rough costs |
| 3.1187 | SOL | okx_pressure;liquidation | source_isolation | isolate_positive_source | positive=okx_pressure;liquidation;negative=l2_imbalance | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| 3.0965 | ALLO | liquidation | source_isolation | repeat_liquidation_not_pressure | positive=liquidation;negative=okx_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| 3.0874 | BSV | hl_candidate | clean_candidate_repeat | repeat_supported_candidate | hl15=0.0017 | repeat the same labels on fresh samples and add rough costs |
| 3.0365 | SAGA | hl_candidate | clean_candidate_repeat | repeat_supported_candidate | hl15=0.0007 | repeat the same labels on fresh samples and add rough costs |
| 2.9792 | PUMP | liquidation;sector_rotation | source_isolation | repeat_liquidation_not_pressure | positive=liquidation;sector_rotation;negative=okx_pressure | repeat sector labels with category membership and costs before mixing with other sources |
| 2.9178 | XLM | okx_pressure;l2_imbalance | source_isolation | isolate_positive_source | positive=okx_pressure;l2_imbalance;negative=liquidation | repeat L2 labels with fill/adverse-selection assumptions before using as directional alpha |
| 2.8356 | HOME | okx_pressure | source_isolation | isolate_positive_source | positive=okx_pressure;negative=liquidation | repeat the positive label source separately |
| 2.7846 | H | liquidation | source_isolation | repeat_liquidation_not_pressure | positive=liquidation;negative=okx_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| 2.5173 | ZRO | hl_candidate | source_isolation | isolate_positive_source | positive=hl_candidate;negative=okx_pressure | repeat the original candidate family and keep unrelated negative sources out of the decision |
| 2.2701 | * | okx_liquidation:short_liquidation_squeeze_watch | family_repeat | repeat_supported_family | cov15=17;mean15=0.00426992;hit15=0.88235294 | collect more labels from this family and compare against neutral/cost baselines |
| 2.1872 | TON | okx_pressure;liquidation | source_isolation | isolate_positive_source | positive=okx_pressure;liquidation;negative=l2_imbalance | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| 2.0283 | BILL | okx_pressure | clean_candidate_repeat | repeat_supported_candidate | okx_pressure15=0.0100 | repeat the same labels on fresh samples and add rough costs |
| 1.9688 | EDEN | okx_pressure | source_isolation | isolate_positive_source | positive=okx_pressure;negative=liquidation | repeat the positive label source separately |
| 1.9270 | PYTH | sector_rotation | clean_candidate_repeat | repeat_supported_candidate | sector15=0.0020:Analytics | repeat the same labels on fresh samples and add rough costs |

## Interpretation

The queue deliberately separates source-specific repeats from cross-lane aggregation. A candidate should graduate only after the same source survives repeated labels and rough cost checks.
