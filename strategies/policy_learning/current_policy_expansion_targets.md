# Current Policy Expansion Targets

This expands paper-supported action preferences into adjacent current lanes. It is not a model, not a strategy implementation, and not a trade list.

| target | seed | context | source | target | action | support | score | decision | next step |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| hype_hype_microstructure_flow_paper_probe_from_near_microstructure_flow_paper_long | near_microstructure_flow_paper_long | microstructure_flow | NEAR | HYPE/hype_microstructure_flow_paper_probe | paper_long | paper_execution_gated | 112.84 | expand_supported_preference_now | open a small paper label for HYPE/hype_microstructure_flow_paper_probe as paper_long, then compare reward to the existing microstructure_flow preference |
| hype_hype_microstructure_flow_probe_from_near_microstructure_flow_paper_long | near_microstructure_flow_paper_long | microstructure_flow | NEAR | HYPE/hype_microstructure_flow_probe | paper_long | paper_1h_supported | 99.04 | expand_supported_preference_now | open a small paper label for HYPE/hype_microstructure_flow_probe as paper_long, then compare reward to the existing microstructure_flow preference |
| eth_eth_microstructure_flow_probe_from_near_microstructure_flow_paper_long | near_microstructure_flow_paper_long | microstructure_flow | NEAR | ETH/eth_microstructure_flow_probe | paper_long | paper_1h_supported | 99.04 | expand_supported_preference_now | open a small paper label for ETH/eth_microstructure_flow_probe as paper_long, then compare reward to the existing microstructure_flow preference |
| sol_solana_stablecoin_migration_from_stablecoin_migration_paper_long | stablecoin_migration_paper_long | stablecoin_migration | family | SOL/solana_stablecoin_migration | paper_long | pending_label | 60.06 | repeat_seed_before_expansion | repeat-label SOL/solana_stablecoin_migration before using this high-reward seed as a broader policy preference |

## Interpretation

A row means a currently observed lane resembles a paper-supported action preference. The next work is to collect new labels and execution evidence, not to hard-code the preference.
