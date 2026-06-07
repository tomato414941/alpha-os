# Current Source Conflict Review

This isolates mixed-evidence candidates by source. It asks which signal source should be repeated or separated next, not whether the asset is deployable.

| asset | score | positives | negatives | action | next test |
| --- | ---: | --- | --- | --- | --- |
| XMR | 3.1004 | hl_candidate | sector_rotation | separate_carry_from_sector | repeat the original candidate family and keep unrelated negative sources out of the decision |
| IP | 2.8166 | hl_candidate | okx_pressure | isolate_positive_source | repeat the original candidate family and keep unrelated negative sources out of the decision |
| BTC | 2.6217 | liquidation;l2_imbalance | okx_pressure | repeat_liquidation_not_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| ONDO | 2.6106 | liquidation;sector_rotation | okx_pressure;l2_imbalance | separate_sector_from_l2 | repeat sector labels with category membership and costs before mixing with other sources |
| ALLO | 2.5965 | liquidation | okx_pressure | repeat_liquidation_not_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| JTO | 2.4579 | liquidation;l2_imbalance | okx_pressure | repeat_liquidation_not_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| XPL | 2.4493 | l2_imbalance;sector_rotation | okx_pressure | repeat_l2_not_pressure | repeat sector labels with category membership and costs before mixing with other sources |
| HOME | 2.3356 | okx_pressure | liquidation | isolate_positive_source | repeat the positive label source separately |
| H | 2.2846 | liquidation | okx_pressure | repeat_liquidation_not_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| SOL | 2.1187 | okx_pressure;liquidation | l2_imbalance | isolate_positive_source | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| ZRO | 2.0173 | hl_candidate | okx_pressure | isolate_positive_source | repeat the original candidate family and keep unrelated negative sources out of the decision |
| PUMP | 1.9792 | liquidation;sector_rotation | okx_pressure | repeat_liquidation_not_pressure | repeat sector labels with category membership and costs before mixing with other sources |
| XLM | 1.9178 | okx_pressure;l2_imbalance | liquidation | isolate_positive_source | repeat L2 labels with fill/adverse-selection assumptions before using as directional alpha |
| EDEN | 1.4688 | okx_pressure | liquidation | isolate_positive_source | repeat the positive label source separately |
| LAB | 1.3363 | liquidation | okx_pressure | repeat_liquidation_not_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| DOGE | 1.3179 | liquidation | okx_pressure;l2_imbalance | repeat_liquidation_not_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| BEAT | 1.2502 | liquidation | okx_pressure | repeat_liquidation_not_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| HYPE | 1.2045 | liquidation | okx_pressure | repeat_liquidation_not_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| TON | 1.1872 | okx_pressure;liquidation | l2_imbalance | isolate_positive_source | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| BNB | 0.8932 | liquidation | okx_pressure | repeat_liquidation_not_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| LINK | 0.7570 | sector_rotation | okx_pressure | isolate_positive_source | repeat sector labels with category membership and costs before mixing with other sources |
| BABY | 0.7532 | okx_pressure | hl_candidate | isolate_positive_source | repeat the positive label source separately |
| JUP | 0.7318 | sector_rotation | hl_candidate | isolate_positive_source | repeat sector labels with category membership and costs before mixing with other sources |
| FARTCOIN | 0.4426 | sector_rotation | okx_pressure | isolate_positive_source | repeat sector labels with category membership and costs before mixing with other sources |
| OPN | 0.3222 | okx_pressure | liquidation | isolate_positive_source | repeat the positive label source separately |

## Interpretation

A mixed candidate is not a failure. It means the project should stop averaging incompatible sources together and repeat the source that is actually carrying the positive label.
