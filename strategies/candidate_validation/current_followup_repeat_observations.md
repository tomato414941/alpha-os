# Current Follow-Up Repeat Observations

This records fresh source-specific observations from the follow-up queue. Each row is asset x source, so mixed evidence is not averaged together before labeling.

| asset | source | source action | dir | priority | mark | funding ann | spread bps | depth 10bps USD | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 | 2.09870000 | 0.109500 | 5.2415 | 11665 | ready_for_label |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 | 2.09870000 | 0.109500 | 5.2415 | 11665 | ready_for_label |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 | 65.52100000 | -0.162707 | 0.1526 | 723180 | ready_for_label |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 | 65.52100000 | -0.162707 | 0.1526 | 723180 | ready_for_label |
| BNB | l2_imbalance | visible_book_imbalance | 1 | 4.8127 | 595.77000000 | 0.109500 | 1.3428 | 88346 | ready_for_label |
| BNB | on_chain_flow | chain_flow_reversal_watch | 1 | 4.8127 | 595.77000000 | 0.109500 | 1.3428 | 88346 | ready_for_label |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 | 1662.80000000 | -0.008981 | 0.6015 | 10621101 | ready_for_label |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 | 0.08163000 | 0.096185 | 3.6753 | 21859 | ready_for_label |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 | 0.74338000 | 0.109500 | 1.6145 | 62554 | ready_for_label |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 | 0.16121000 | -0.039375 | 0.6205 | 50204 | ready_for_label |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 | 0.66170000 | -0.159211 | 1.5114 | 16741 | ready_for_label |
| BTC | on_chain_flow | chain_flow_reversal_watch | 1 | 3.0455 | 62922.00000000 | 0.092945 | 0.1589 | 3663105 | ready_for_label |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 | 62.46700000 | 0.109500 | 0.3202 | 113501 | ready_for_label |
| XMR | l2_imbalance | visible_book_imbalance | 1 | 2.3874 | 302.40000000 | 0.485862 | 0.9924 | 8004 | ready_for_label |
| WLD | l2_imbalance | visible_book_imbalance | 1 | 2.3679 | 0.47360000 | 0.109500 | 3.5875 | 10352 | ready_for_label |

## Interpretation

`ready_for_label` means the source had a direction and can be labeled after 15m/1h. `missing_source_direction` keeps the context visible but does not create a directional alpha label.
