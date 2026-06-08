# Current Source Conflict Review

This isolates mixed-evidence candidates by source. It asks which signal source should be repeated or separated next, not whether the asset is deployable.

| asset | score | positives | negatives | action | next test |
| --- | ---: | --- | --- | --- | --- |
| MEGA | 3.4940 | exchange_catalyst;on_chain_flow | exchange_catalyst | isolate_positive_source | repeat the positive label source separately |
| ADA | 2.9842 | on_chain_flow | l2_imbalance | isolate_positive_source | repeat the positive label source separately |
| BTC | 2.5455 | on_chain_flow | liquidation | isolate_positive_source | repeat the positive label source separately |
| HYPE | 2.1935 | on_chain_flow | liquidation | isolate_positive_source | repeat the positive label source separately |
| POL | 1.7572 | sector_perp_context;on_chain_flow | exchange_catalyst | isolate_positive_source | repeat the positive label source separately |

## Interpretation

A mixed candidate is not a failure. It means the project should stop averaging incompatible sources together and repeat the source that is actually carrying the positive label.
