# Current Follow-Up Repeat Forward Labels

This labels source-specific follow-up observations against subsequent Hyperliquid returns. It is a repeat-observation label, not a PnL model.

| asset | source | action | dir | priority | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 | 0.007706 | 0.007706 |  |  | labeled_15m_pending_1h |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 | 0.007660 | 0.007660 |  |  | labeled_15m_pending_1h |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 | 0.007660 | 0.007660 |  |  | labeled_15m_pending_1h |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 | 0.007603 | 0.007603 |  |  | labeled_15m_pending_1h |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 | 0.004854 | 0.004854 |  |  | labeled_15m_pending_1h |
| MON | on_chain_flow | chain_flow_reversal_watch | 1 | 3.8131 | 0.004495 | 0.004495 |  |  | labeled_15m_pending_1h |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 | 0.004478 | 0.004478 |  |  | labeled_15m_pending_1h |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 | 0.003124 | 0.003124 |  |  | labeled_15m_pending_1h |
| XMR | l2_imbalance | visible_book_imbalance | 1 | 2.3874 | 0.002281 | 0.002281 |  |  | labeled_15m_pending_1h |
| BTC | on_chain_flow | chain_flow_reversal_watch | 1 | 3.0455 | 0.001961 | 0.001961 |  |  | labeled_15m_pending_1h |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 | 0.001844 | 0.001844 |  |  | labeled_15m_pending_1h |
| BNB | l2_imbalance | visible_book_imbalance | 1 | 4.8127 | 0.000840 | 0.000840 |  |  | labeled_15m_pending_1h |
| BNB | on_chain_flow | chain_flow_reversal_watch | 1 | 4.8127 | 0.000840 | 0.000840 |  |  | labeled_15m_pending_1h |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 | 0.000622 | 0.000622 |  |  | labeled_15m_pending_1h |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 | 0.003124 | -0.003124 |  |  | labeled_15m_pending_1h |

## Interpretation

`pending_15m` or `pending_1h` means the observation has not matured. Positive directional return means the source-specific direction was right over that horizon before fees, funding PnL, and slippage.
