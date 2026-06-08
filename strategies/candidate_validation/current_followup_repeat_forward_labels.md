# Current Follow-Up Repeat Forward Labels

This labels source-specific follow-up observations against subsequent Hyperliquid returns. It is a repeat-observation label, not a PnL model.

| asset | source | action | dir | priority | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 |  |  |  |  | pending_15m |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 |  |  |  |  | pending_15m |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 |  |  |  |  | pending_15m |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 |  |  |  |  | pending_15m |
| BNB | l2_imbalance | visible_book_imbalance | 1 | 4.8127 |  |  |  |  | pending_15m |
| BNB | on_chain_flow | chain_flow_reversal_watch | 1 | 4.8127 |  |  |  |  | pending_15m |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 |  |  |  |  | pending_15m |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 |  |  |  |  | pending_15m |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 |  |  |  |  | pending_15m |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 |  |  |  |  | pending_15m |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 |  |  |  |  | pending_15m |
| BTC | on_chain_flow | chain_flow_reversal_watch | 1 | 3.0455 |  |  |  |  | pending_15m |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 |  |  |  |  | pending_15m |
| WLD | l2_imbalance | visible_book_imbalance | 1 | 2.3679 |  |  |  |  | pending_15m |

## Interpretation

`pending_15m` or `pending_1h` means the observation has not matured. Positive directional return means the source-specific direction was right over that horizon before fees, funding PnL, and slippage.
