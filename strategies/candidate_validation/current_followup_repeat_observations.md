# Current Follow-Up Repeat Observations

This records fresh source-specific observations from the follow-up queue. Each row is asset x source, so mixed evidence is not averaged together before labeling.

| asset | source | source action | dir | priority | mark | funding ann | spread bps | depth 10bps USD | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 | 2.17640000 | 0.109500 | 2.7527 | 57413 | ready_for_label |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 | 2.17640000 | 0.109500 | 2.7527 | 57413 | ready_for_label |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 | 65.73700000 | -0.316550 | 0.1521 | 224820 | ready_for_label |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 | 65.73700000 | -0.316550 | 0.1521 | 224820 | ready_for_label |
| BNB | l2_imbalance | visible_book_imbalance | 1 | 4.8127 | 595.57000000 | 0.109500 | 1.1751 | 86335 | ready_for_label |
| BNB | on_chain_flow | chain_flow_reversal_watch | 1 | 4.8127 | 595.57000000 | 0.109500 | 1.1751 | 86335 | ready_for_label |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 | 1669.10000000 | -0.000950 | 0.5989 | 12958184 | ready_for_label |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 | 0.08186000 | 0.006147 | 3.6623 | 10300 | ready_for_label |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 | 0.74767000 | 0.109500 | 0.1337 | 75354 | ready_for_label |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 | 0.16197000 | 0.036465 | 0.6170 | 65229 | ready_for_label |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 | 0.66500000 | -0.204665 | 1.5032 | 13768 | ready_for_label |
| BTC | on_chain_flow | chain_flow_reversal_watch | 1 | 3.0455 | 63118.00000000 | 0.020223 | 0.1584 | 2407309 | ready_for_label |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 | 63.04700000 | 0.109500 | 0.1586 | 110716 | ready_for_label |
| WLD | l2_imbalance | visible_book_imbalance | 1 | 2.3679 | 0.47929000 | 0.109500 | 1.8773 | 22759 | ready_for_label |

## Interpretation

`ready_for_label` means the source had a direction and can be labeled after 15m/1h. `missing_source_direction` keeps the context visible but does not create a directional alpha label.
