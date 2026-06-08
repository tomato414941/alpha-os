# Current Follow-Up Repeat Forward Labels

This labels source-specific follow-up observations against subsequent Hyperliquid returns. It is a repeat-observation label, not a PnL model.

| asset | source | action | dir | priority | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 | 0.006654 | 0.006654 |  |  | labeled_15m_pending_1h |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 | 0.001827 | 0.001827 |  |  | labeled_15m_pending_1h |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 | 0.001786 | 0.001786 |  |  | labeled_15m_pending_1h |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 | 0.000956 | 0.000956 |  |  | labeled_15m_pending_1h |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 | -0.000319 | 0.000319 |  |  | labeled_15m_pending_1h |
| BNB | l2_imbalance | visible_book_imbalance | 1 | 4.8127 | 0.000218 | 0.000218 |  |  | labeled_15m_pending_1h |
| BNB | on_chain_flow | chain_flow_reversal_watch | 1 | 4.8127 | 0.000218 | 0.000218 |  |  | labeled_15m_pending_1h |
| BTC | on_chain_flow | chain_flow_reversal_watch | 1 | 3.0455 | 0.000111 | 0.000111 |  |  | labeled_15m_pending_1h |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 | -0.000150 | -0.000150 |  |  | labeled_15m_pending_1h |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 | -0.000319 | -0.000319 |  |  | labeled_15m_pending_1h |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 | -0.001604 | -0.001604 |  |  | labeled_15m_pending_1h |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 | -0.001604 | -0.001604 |  |  | labeled_15m_pending_1h |
| WLD | l2_imbalance | visible_book_imbalance | 1 | 2.3679 | -0.002833 | -0.002833 |  |  | labeled_15m_pending_1h |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 | -0.003107 | -0.003107 |  |  | labeled_15m_pending_1h |

## Interpretation

`pending_15m` or `pending_1h` means the observation has not matured. Positive directional return means the source-specific direction was right over that horizon before fees, funding PnL, and slippage.
