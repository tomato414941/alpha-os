# Current Follow-Up OKX Repeat Forward Labels

This labels OKX source-specific follow-up observations against OKX 15m candles. It is not net PnL.

| asset | source | action | dir | priority | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MEGA | exchange_catalyst | spot_listing_watch | 1 | 4.4940 | 0.004220 | 0.004220 |  |  | labeled_15m_pending_1h |
| WLD | l2_imbalance | visible_book_imbalance | 1 | 2.3679 | 0.001047 | 0.001047 |  |  | labeled_15m_pending_1h |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 | 0.000940 | 0.000940 |  |  | labeled_15m_pending_1h |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 | 0.000940 | 0.000940 |  |  | labeled_15m_pending_1h |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 | 0.000537 | 0.000537 |  |  | labeled_15m_pending_1h |
| SEI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.8994 | 0.000203 | 0.000203 |  |  | labeled_15m_pending_1h |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 | -0.000294 | -0.000294 |  |  | labeled_15m_pending_1h |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 | -0.000600 | -0.000600 |  |  | labeled_15m_pending_1h |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 | -0.000609 | -0.000609 |  |  | labeled_15m_pending_1h |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 | -0.000617 | -0.000617 |  |  | labeled_15m_pending_1h |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 | -0.000957 | -0.000957 |  |  | labeled_15m_pending_1h |
| MEGA | on_chain_flow | chain_outflow_stress_watch | -1 | 4.4940 | 0.004220 | -0.004220 |  |  | labeled_15m_pending_1h |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 | 0.000000 | -0.000000 |  |  | labeled_15m_pending_1h |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 | 0.000000 | 0.000000 |  |  | labeled_15m_pending_1h |
| CHIP | exchange_catalyst | spot_listing_watch | 1 | 3.9957 | 0.000000 | 0.000000 |  |  | labeled_15m_pending_1h |
| POL | sector_perp_context | sector_momentum_watch | 1 | 2.7572 | 0.000000 | 0.000000 |  |  | labeled_15m_pending_1h |
| POL | on_chain_flow | chain_flow_reversal_watch | 1 | 2.7572 | 0.000000 | 0.000000 |  |  | labeled_15m_pending_1h |
| PEPE | liquidation | long_liquidation_cascade_watch | -1 | 2.1830 | 0.000000 | -0.000000 |  |  | labeled_15m_pending_1h |

## Interpretation

Pending rows have not matured. Positive directional return means the source-specific direction was right on OKX before fees and slippage.
