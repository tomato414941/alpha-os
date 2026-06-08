# Current Follow-Up Repeat Forward Labels

This labels source-specific follow-up observations against subsequent Hyperliquid returns. It is a repeat-observation label, not a PnL model.

| asset | source | action | dir | priority | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 | 0.001128 | 0.001128 |  |  | labeled_15m_pending_1h |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 | 0.001128 | 0.001128 |  |  | labeled_15m_pending_1h |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 | 0.000711 | 0.000711 |  |  | labeled_15m_pending_1h |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 | -0.000593 | 0.000593 |  |  | labeled_15m_pending_1h |
| BTC | on_chain_flow | chain_flow_reversal_watch | 1 | 3.0455 | -0.000301 | -0.000301 |  |  | labeled_15m_pending_1h |
| BNB | l2_imbalance | visible_book_imbalance | 1 | 4.8127 | -0.000352 | -0.000352 |  |  | labeled_15m_pending_1h |
| BNB | on_chain_flow | chain_flow_reversal_watch | 1 | 4.8127 | -0.000352 | -0.000352 |  |  | labeled_15m_pending_1h |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 | -0.000540 | -0.000540 |  |  | labeled_15m_pending_1h |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 | -0.000593 | -0.000593 |  |  | labeled_15m_pending_1h |
| WLD | l2_imbalance | visible_book_imbalance | 1 | 2.3679 | -0.000711 | -0.000711 |  |  | labeled_15m_pending_1h |
| XMR | l2_imbalance | visible_book_imbalance | 1 | 2.3874 | -0.000820 | -0.000820 |  |  | labeled_15m_pending_1h |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 | -0.000900 | -0.000900 |  |  | labeled_15m_pending_1h |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 | -0.001096 | -0.001096 |  |  | labeled_15m_pending_1h |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 | -0.002088 | -0.002088 |  |  | labeled_15m_pending_1h |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 | 0.000000 | 0.000000 |  |  | labeled_15m_pending_1h |

## Interpretation

`pending_15m` or `pending_1h` means the observation has not matured. Positive directional return means the source-specific direction was right over that horizon before fees, funding PnL, and slippage.
