# Current Follow-Up OKX Repeat Forward Labels

This labels OKX source-specific follow-up observations against OKX 15m candles. It is not net PnL.

| asset | source | action | dir | priority | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| WLD | l2_imbalance | visible_book_imbalance | 1 | 2.3679 | 0.019688 | 0.019688 |  |  | labeled_15m_pending_1h |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 | 0.008130 | 0.008130 |  |  | labeled_15m_pending_1h |
| MEGA | exchange_catalyst | spot_listing_watch | 1 | 4.4940 | 0.007774 | 0.007774 |  |  | labeled_15m_pending_1h |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 | 0.007073 | 0.007073 |  |  | labeled_15m_pending_1h |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 | 0.006750 | 0.006750 |  |  | labeled_15m_pending_1h |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 | 0.006750 | 0.006750 |  |  | labeled_15m_pending_1h |
| CHIP | exchange_catalyst | spot_listing_watch | 1 | 3.9957 | 0.004605 | 0.004605 |  |  | labeled_15m_pending_1h |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 | 0.004547 | 0.004547 |  |  | labeled_15m_pending_1h |
| SEI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.8994 | 0.004515 | 0.004515 |  |  | labeled_15m_pending_1h |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 | 0.004350 | 0.004350 |  |  | labeled_15m_pending_1h |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 | 0.003370 | 0.003370 |  |  | labeled_15m_pending_1h |
| POL | sector_perp_context | sector_momentum_watch | 1 | 2.7572 | 0.002918 | 0.002918 |  |  | labeled_15m_pending_1h |
| POL | on_chain_flow | chain_flow_reversal_watch | 1 | 2.7572 | 0.002918 | 0.002918 |  |  | labeled_15m_pending_1h |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 | 0.001868 | 0.001868 |  |  | labeled_15m_pending_1h |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 | 0.000737 | 0.000737 |  |  | labeled_15m_pending_1h |
| PEPE | liquidation | long_liquidation_cascade_watch | -1 | 2.1830 | 0.002899 | -0.002899 |  |  | labeled_15m_pending_1h |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 | 0.003370 | -0.003370 |  |  | labeled_15m_pending_1h |
| MEGA | on_chain_flow | chain_outflow_stress_watch | -1 | 4.4940 | 0.007774 | -0.007774 |  |  | labeled_15m_pending_1h |

## Interpretation

Pending rows have not matured. Positive directional return means the source-specific direction was right on OKX before fees and slippage.
