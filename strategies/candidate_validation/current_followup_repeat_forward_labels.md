# Current Follow-Up Repeat Forward Labels

This labels source-specific follow-up observations against subsequent Hyperliquid returns. It is a repeat-observation label, not a PnL model.

| asset | source | action | dir | priority | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 | 0.007428 | 0.007428 | 0.016689 | 0.016689 | labeled_1h |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 | 0.007428 | 0.007428 | 0.016689 | 0.016689 | labeled_1h |
| XMR | l2_imbalance | visible_book_imbalance | 1 | 2.3874 | 0.004133 | 0.004133 | 0.015415 | 0.015415 | labeled_1h |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 | 0.003969 | 0.003969 | 0.003889 | 0.003889 | labeled_1h |
| WLD | l2_imbalance | visible_book_imbalance | 1 | 2.3679 | 0.003326 | 0.003326 | 0.001339 | 0.001339 | labeled_1h |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 | 0.001945 | 0.001945 | -0.001838 | -0.001838 | labeled_1h |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 | -0.000380 | 0.000380 | -0.002769 | 0.002769 | labeled_1h |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 | -0.000380 | -0.000380 | -0.002769 | -0.002769 | labeled_1h |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 | -0.000720 | -0.000720 | -0.000900 | -0.000900 | labeled_1h |
| BTC | on_chain_flow | chain_flow_reversal_watch | 1 | 3.0455 | -0.000809 | -0.000809 | -0.000841 | -0.000841 | labeled_1h |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 | -0.001172 | -0.001172 | -0.003516 | -0.003516 | labeled_1h |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 | -0.001583 | -0.001583 | -0.006821 | -0.006821 | labeled_1h |
| BNB | l2_imbalance | visible_book_imbalance | 1 | 4.8127 | -0.001709 | -0.001709 | -0.002798 | -0.002798 | labeled_1h |
| BNB | on_chain_flow | chain_flow_reversal_watch | 1 | 4.8127 | -0.001709 | -0.001709 | -0.002798 | -0.002798 | labeled_1h |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 | -0.002999 | -0.002999 | -0.007196 | -0.007196 | labeled_1h |

## Interpretation

`pending_15m` or `pending_1h` means the observation has not matured. Positive directional return means the source-specific direction was right over that horizon before fees, funding PnL, and slippage.
