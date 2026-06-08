# Current Follow-Up OKX Repeat Observations

This records source-specific OKX observations from the follow-up queue. It keeps OKX-only candidates visible for later labels.

| asset | source | source action | dir | priority | inst | last | funding ann | spread bps | depth 10bps USD | status |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| NEAR | exchange_catalyst | network_event_watch | 1 | 8.3498 | NEAR-USDT-SWAP | 2.02600000 | 0.298995 | 4.9322 | 10890 | ready_for_label |
| NEAR | on_chain_flow | chain_flow_reversal_watch | 1 | 8.3498 | NEAR-USDT-SWAP | 2.02600000 | 0.298995 | 4.9322 | 10890 | ready_for_label |
| SOL | exchange_catalyst | exchange_removal_watch | -1 | 5.2019 | SOL-USDT-SWAP | 65.11000000 | -1.874607 | 1.5360 | 967360 | ready_for_label |
| SOL | on_chain_flow | chain_flow_reversal_watch | 1 | 5.2019 | SOL-USDT-SWAP | 65.11000000 | -1.874607 | 1.5360 | 967360 | ready_for_label |
| MEGA | exchange_catalyst | spot_listing_watch | 1 | 4.4940 | MEGA-USDT-SWAP | 0.04789000 | -0.630294 | 2.0892 | 4609 | ready_for_label |
| MEGA | on_chain_flow | chain_outflow_stress_watch | -1 | 4.4940 | MEGA-USDT-SWAP | 0.04789000 | -0.630294 | 2.0892 | 4609 | ready_for_label |
| CHIP | exchange_catalyst | spot_listing_watch | 1 | 3.9957 | CHIP-USDT-SWAP | 0.03026000 | -1.637993 | 3.3063 | 5286 | ready_for_label |
| ETH | on_chain_flow | chain_flow_reversal_watch | 1 | 3.9347 | ETH-USDT-SWAP | 1647.73000000 | -0.453727 | 0.0607 | 1098579 | ready_for_label |
| SEI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.8994 | SEI-USDT-SWAP | 0.04848000 | -1.031922 | 2.0629 | 5879 | ready_for_label |
| ARB | on_chain_flow | chain_flow_reversal_watch | 1 | 3.7070 | ARB-USDT-SWAP | 0.08102000 | -0.363761 | 1.2343 | 18154 | ready_for_label |
| SUI | on_chain_flow | chain_flow_reversal_watch | 1 | 3.6307 | SUI-USDT-SWAP | 0.73160000 | 0.075988 | 1.3670 | 69797 | ready_for_label |
| ADA | on_chain_flow | chain_flow_reversal_watch | 1 | 3.4842 | ADA-USDT-SWAP | 0.16030000 | -0.355626 | 6.2402 | 98169 | ready_for_label |
| APT | on_chain_flow | chain_flow_reversal_watch | 1 | 3.3984 | APT-USDT-SWAP | 0.65600000 | -0.985826 | 1.5245 | 16534 | ready_for_label |
| POL | sector_perp_context | sector_momentum_watch | 1 | 2.7572 | POL-USDT-SWAP | 0.07858000 | 0.438000 | 1.2727 | 9908 | ready_for_label |
| POL | on_chain_flow | chain_flow_reversal_watch | 1 | 2.7572 | POL-USDT-SWAP | 0.07858000 | 0.438000 | 1.2727 | 9908 | ready_for_label |
| HYPE | on_chain_flow | chain_flow_reversal_watch | 1 | 2.6935 | HYPE-USDT-SWAP | 61.04000000 | -1.188548 | 1.6381 | 232013 | ready_for_label |
| WLD | l2_imbalance | visible_book_imbalance | 1 | 2.3679 | WLD-USDT-SWAP | 0.47030000 | -1.687482 | 2.1261 | 45230 | ready_for_label |
| PEPE | liquidation | long_liquidation_cascade_watch | -1 | 2.1830 | PEPE-USDT-SWAP | 0.00000275 | -0.463955 | 3.6370 | 142169 | ready_for_label |

## Interpretation

These rows should be labeled on OKX candles after 15m/1h. Positive or negative outcomes should be compared by source, not only by asset.
