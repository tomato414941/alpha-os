# Current Follow-Up OKX Repeat Observations

This records source-specific OKX observations from the follow-up queue. It keeps OKX-only candidates visible for later labels.

| asset | source | source action | dir | priority | inst | last | funding ann | spread bps | depth 10bps USD | status |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 | NEAR-USDT-SWAP | 2.17900000 | 0.003965 | 4.5882 | 33699 | ready_for_label |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 | NEAR-USDT-SWAP | 2.17900000 | 0.003965 | 4.5882 | 33699 | ready_for_label |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 | SOL-USDT-SWAP | 65.77000000 | -1.758454 | 1.5206 | 940663 | ready_for_label |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 | SOL-USDT-SWAP | 65.77000000 | -1.758454 | 1.5206 | 940663 | ready_for_label |
| MEGA | exchange_catalyst | spot_listing_watch | 1 | 4.4940 | MEGA-USDT-SWAP | 0.04928000 | -1.634342 | 2.0294 | 8439 | ready_for_label |
| MEGA | on_chain_flow | chain_outflow_stress_watch | -1 | 4.4940 | MEGA-USDT-SWAP | 0.04928000 | -1.634342 | 2.0294 | 8439 | ready_for_label |
| CHIP | exchange_catalyst | spot_listing_watch | 1 | 3.9957 | CHIP-USDT-SWAP | 0.03086000 | -0.288352 | 3.2410 | 6075 | ready_for_label |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 | ETH-USDT-SWAP | 1669.64000000 | -0.596208 | 0.0599 | 1391452 | ready_for_label |
| SEI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.8994 | SEI-USDT-SWAP | 0.04923000 | -0.278966 | 2.0307 | 8764 | ready_for_label |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 | ARB-USDT-SWAP | 0.08193000 | 0.433148 | 1.2206 | 20301 | ready_for_label |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 | SUI-USDT-SWAP | 0.74750000 | -0.232266 | 1.3379 | 104996 | ready_for_label |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 | ADA-USDT-SWAP | 0.16190000 | -0.557211 | 6.1747 | 78882 | ready_for_label |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 | APT-USDT-SWAP | 0.66530000 | -0.491273 | 1.5032 | 19642 | ready_for_label |
| STRK | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3542 | STRK-USDT-SWAP | 0.03363000 | 0.438000 | 2.9731 | 4028 | ready_for_label |
| POL | sector_perp_context | sector_momentum_watch | 1 | 2.7572 | POL-USDT-SWAP | 0.07906000 | 0.349817 | 1.2649 | 10258 | ready_for_label |
| POL | on_chain_flow | chain_flow_reversal_watch | 1 | 2.7572 | POL-USDT-SWAP | 0.07906000 | 0.349817 | 1.2649 | 10258 | ready_for_label |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 | HYPE-USDT-SWAP | 63.03000000 | -0.411996 | 1.5867 | 273559 | ready_for_label |
| WLD | l2_imbalance | visible_book_imbalance | 1 | 2.3679 | WLD-USDT-SWAP | 0.47770000 | -1.313766 | 2.0931 | 42748 | ready_for_label |
| PEPE | liquidation | long_liquidation_cascade_watch | -1 | 2.1830 | PEPE-USDT-SWAP | 0.00000278 | -0.168587 | 3.5913 | 127762 | ready_for_label |

## Interpretation

These rows should be labeled on OKX candles after 15m/1h. Positive or negative outcomes should be compared by source, not only by asset.
