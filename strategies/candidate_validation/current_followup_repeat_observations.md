# Current Follow-Up Repeat Observations

This records fresh source-specific observations from the follow-up queue. Each row is asset x source, so mixed evidence is not averaged together before labeling.

| asset | source | source action | dir | priority | mark | funding ann | spread bps | depth 10bps USD | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 | 2.02970000 | 0.109500 | 1.9692 | 41222 | ready_for_label |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 | 2.02970000 | 0.109500 | 1.9692 | 41222 | ready_for_label |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 | 65.14800000 | -0.273009 | 0.3069 | 540694 | ready_for_label |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 | 65.14800000 | -0.273009 | 0.3069 | 540694 | ready_for_label |
| BNB | l2_imbalance | visible_book_imbalance | 1 | 4.8127 | 592.65000000 | 0.109500 | 2.0236 | 142635 | ready_for_label |
| BNB | on_chain_flow | chain_flow_reversal_watch | 1 | 4.8127 | 592.65000000 | 0.109500 | 2.0236 | 142635 | ready_for_label |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 | 1648.70000000 | -0.107213 | 0.6063 | 10872720 | ready_for_label |
| MON | on_chain_flow | chain_flow_reversal_watch | 1 | 3.8131 | 0.02147400 | -0.168836 | 5.5863 | 5503 | ready_for_label |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 | 0.08100000 | 0.091928 | 2.4679 | 18293 | ready_for_label |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 | 0.73167000 | 0.002876 | 2.3229 | 88623 | ready_for_label |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 | 0.16010000 | -0.188968 | 0.6240 | 38037 | ready_for_label |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 | 0.65570000 | -0.104895 | 4.5728 | 12039 | ready_for_label |
| BTC | on_chain_flow | chain_flow_reversal_watch | 1 | 3.0455 | 62517.00000000 | 0.031727 | 0.1599 | 4131924 | ready_for_label |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 | 60.91000000 | 0.109500 | 3.4447 | 145397 | ready_for_label |
| XMR | l2_imbalance | visible_book_imbalance | 1 | 2.3874 | 301.34000000 | 0.417327 | 1.3267 | 7390 | ready_for_label |

## Interpretation

`ready_for_label` means the source had a direction and can be labeled after 15m/1h. `missing_source_direction` keeps the context visible but does not create a directional alpha label.
