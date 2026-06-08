# Current Follow-Up OKX Repeat Observations

This records source-specific OKX observations from the follow-up queue. It keeps OKX-only candidates visible for later labels.

| asset | source | source action | dir | priority | inst | last | funding ann | spread bps | depth 10bps USD | status |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 | NEAR-USDT-SWAP | 2.09800000 | 0.304239 | 4.7653 | 15209 | ready_for_label |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 | NEAR-USDT-SWAP | 2.09800000 | 0.304239 | 4.7653 | 15209 | ready_for_label |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 | SOL-USDT-SWAP | 65.50000000 | -1.882832 | 1.5266 | 893460 | ready_for_label |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 | SOL-USDT-SWAP | 65.50000000 | -1.882832 | 1.5266 | 893460 | ready_for_label |
| MEGA | exchange_catalyst | spot_listing_watch | 1 | 4.4940 | MEGA-USDT-SWAP | 0.04978000 | -0.751603 | 2.0090 | 5398 | ready_for_label |
| MEGA | on_chain_flow | chain_outflow_stress_watch | -1 | 4.4940 | MEGA-USDT-SWAP | 0.04978000 | -0.751603 | 2.0090 | 5398 | ready_for_label |
| CHIP | exchange_catalyst | spot_listing_watch | 1 | 3.9957 | CHIP-USDT-SWAP | 0.03063000 | -1.136071 | 3.2642 | 6491 | ready_for_label |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 | ETH-USDT-SWAP | 1662.88000000 | -0.464208 | 0.0601 | 1294241 | ready_for_label |
| SEI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.8994 | SEI-USDT-SWAP | 0.04900000 | -0.874780 | 2.0406 | 7852 | ready_for_label |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 | ARB-USDT-SWAP | 0.08163000 | -0.067995 | 1.2250 | 14952 | ready_for_label |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 | SUI-USDT-SWAP | 0.74290000 | 0.010479 | 1.3460 | 94796 | ready_for_label |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 | ADA-USDT-SWAP | 0.16110000 | -0.392665 | 6.2054 | 90793 | ready_for_label |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 | APT-USDT-SWAP | 0.66170000 | -0.647906 | 1.5109 | 14249 | ready_for_label |
| POL | sector_perp_context | sector_momentum_watch | 1 | 2.7572 | POL-USDT-SWAP | 0.07906000 | 0.438000 | 1.2649 | 9090 | ready_for_label |
| POL | on_chain_flow | chain_flow_reversal_watch | 1 | 2.7572 | POL-USDT-SWAP | 0.07906000 | 0.438000 | 1.2649 | 9090 | ready_for_label |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 | HYPE-USDT-SWAP | 62.54000000 | -0.943055 | 1.6001 | 311174 | ready_for_label |
| WLD | l2_imbalance | visible_book_imbalance | 1 | 2.3679 | WLD-USDT-SWAP | 0.47440000 | -1.741737 | 2.1081 | 33101 | ready_for_label |
| PEPE | liquidation | long_liquidation_cascade_watch | -1 | 2.1830 | PEPE-USDT-SWAP | 0.00000277 | -0.295779 | 3.6095 | 143638 | ready_for_label |

## Interpretation

These rows should be labeled on OKX candles after 15m/1h. Positive or negative outcomes should be compared by source, not only by asset.
