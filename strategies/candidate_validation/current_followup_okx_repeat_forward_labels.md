# Current Follow-Up OKX Repeat Forward Labels

This labels OKX source-specific follow-up observations against OKX 15m candles. It is not net PnL.

| asset | source | action | dir | priority | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 | 0.007052 | 0.007052 | 0.014104 | 0.014104 | labeled_1h |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 | 0.007052 | 0.007052 | 0.014104 | 0.014104 | labeled_1h |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 | 0.004307 | 0.004307 | 0.003828 | 0.003828 | labeled_1h |
| WLD | l2_imbalance | visible_book_imbalance | 1 | 2.3679 | 0.003351 | 0.003351 | 0.003561 | 0.003561 | labeled_1h |
| MEGA | exchange_catalyst | spot_listing_watch | 1 | 4.4940 | 0.002613 | 0.002613 | -0.013867 | -0.013867 | labeled_1h |
| CHIP | exchange_catalyst | spot_listing_watch | 1 | 3.9957 | 0.001946 | 0.001946 | -0.000324 | -0.000324 | labeled_1h |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 | 0.001879 | 0.001879 | -0.001611 | -0.001611 | labeled_1h |
| PEPE | liquidation | long_liquidation_cascade_watch | -1 | 2.1830 | -0.001438 | 0.001438 | -0.002516 | 0.002516 | labeled_1h |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 | -0.000304 | 0.000304 | -0.002587 | 0.002587 | labeled_1h |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 | -0.000304 | -0.000304 | -0.002587 | -0.002587 | labeled_1h |
| SEI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.8994 | -0.000406 | -0.000406 | -0.006297 | -0.006297 | labeled_1h |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 | -0.000420 | -0.000420 | -0.000648 | -0.000648 | labeled_1h |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 | -0.001234 | -0.001234 | -0.004318 | -0.004318 | labeled_1h |
| POL | sector_perp_context | sector_momentum_watch | 1 | 2.7572 | -0.001263 | -0.001263 | -0.006692 | -0.006692 | labeled_1h |
| POL | on_chain_flow | chain_flow_reversal_watch | 1 | 2.7572 | -0.001263 | -0.001263 | -0.006692 | -0.006692 | labeled_1h |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 | -0.001583 | -0.001583 | -0.006698 | -0.006698 | labeled_1h |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 | -0.002549 | -0.002549 | -0.007197 | -0.007197 | labeled_1h |
| MEGA | on_chain_flow | chain_outflow_stress_watch | -1 | 4.4940 | 0.002613 | -0.002613 | -0.013867 | 0.013867 | labeled_1h |

## Interpretation

Pending rows have not matured. Positive directional return means the source-specific direction was right on OKX before fees and slippage.
