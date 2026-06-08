# Current Follow-Up OKX Repeat Forward Labels

This labels OKX source-specific follow-up observations against OKX 15m candles. It is not net PnL.

| asset | source | action | dir | priority | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 | 0.006516 | 0.006516 |  |  | labeled_15m_pending_1h |
| SEI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.8994 | 0.002434 | 0.002434 |  |  | labeled_15m_pending_1h |
| CHIP | exchange_catalyst | spot_listing_watch | 1 | 3.9957 | 0.002260 | 0.002260 |  |  | labeled_15m_pending_1h |
| MEGA | on_chain_flow | chain_outflow_stress_watch | -1 | 4.4940 | -0.001606 | 0.001606 |  |  | labeled_15m_pending_1h |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 | 0.001233 | 0.001233 |  |  | labeled_15m_pending_1h |
| STRK | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3542 | 0.001187 | 0.001187 |  |  | labeled_15m_pending_1h |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 | 0.001004 | 0.001004 |  |  | labeled_15m_pending_1h |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 | 0.000974 | 0.000974 |  |  | labeled_15m_pending_1h |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 | -0.000152 | 0.000152 |  |  | labeled_15m_pending_1h |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 | -0.000150 | -0.000150 |  |  | labeled_15m_pending_1h |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 | -0.000152 | -0.000152 |  |  | labeled_15m_pending_1h |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 | -0.000917 | -0.000917 |  |  | labeled_15m_pending_1h |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 | -0.000917 | -0.000917 |  |  | labeled_15m_pending_1h |
| PEPE | liquidation | long_liquidation_cascade_watch | -1 | 2.1830 | 0.001435 | -0.001435 |  |  | labeled_15m_pending_1h |
| MEGA | exchange_catalyst | spot_listing_watch | 1 | 4.4940 | -0.001606 | -0.001606 |  |  | labeled_15m_pending_1h |
| POL | sector_perp_context | sector_momentum_watch | 1 | 2.7572 | -0.002390 | -0.002390 |  |  | labeled_15m_pending_1h |
| POL | on_chain_flow | chain_flow_reversal_watch | 1 | 2.7572 | -0.002390 | -0.002390 |  |  | labeled_15m_pending_1h |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 | -0.002697 | -0.002697 |  |  | labeled_15m_pending_1h |
| WLD | l2_imbalance | visible_book_imbalance | 1 | 2.3679 | -0.002959 | -0.002959 |  |  | labeled_15m_pending_1h |

## Interpretation

Pending rows have not matured. Positive directional return means the source-specific direction was right on OKX before fees and slippage.
