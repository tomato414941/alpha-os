# Current Follow-Up OKX Execution Context

This joins follow-up candidates to current OKX USDT swap ticker, funding, spread, and public book depth. It is a rough venue context, not a fill model.

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | NEAR-USDT-SWAP | exchange_catalyst;on_chain_flow | 8.3498 | 0.876000 | 74368805 | 4.6479 | 38097 | 0.026249 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SOL | SOL-USDT-SWAP | exchange_catalyst;on_chain_flow | 5.2019 | -1.462892 | 11131083 | 1.5180 | 983057 | 0.001017 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MEGA | MEGA-USDT-SWAP | exchange_catalyst;on_chain_flow | 4.4940 | -4.878198 | 243112010 | 1.9495 | 5137 | 0.194648 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ETH | ETH-USDT-SWAP | on_chain_flow | 3.9347 | -0.447243 | 4937649 | 0.0602 | 1332013 | 0.000751 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SEI | SEI-USDT-SWAP | on_chain_flow | 3.8994 | -0.311891 | 66590340 | 2.0119 | 9266 | 0.107923 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ARB | ARB-USDT-SWAP | on_chain_flow | 3.7070 | 0.827621 | 133495269 | 1.2248 | 16926 | 0.059080 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SUI | SUI-USDT-SWAP | on_chain_flow | 3.6307 | -0.217097 | 111898657 | 1.3329 | 94763 | 0.010553 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ADA | ADA-USDT-SWAP | on_chain_flow | 3.4842 | 0.367251 | 408285830 | 6.1106 | 88126 | 0.011347 | okx_context_ok | OKX public context does not obviously block a small repeat |
| APT | APT-USDT-SWAP | on_chain_flow | 3.3984 | 0.440679 | 17600653 | 1.5114 | 18904 | 0.052899 | okx_context_ok | OKX public context does not obviously block a small repeat |
| STRK | STRK-USDT-SWAP | on_chain_flow | 3.3542 | 0.438000 | 105952841 | 2.9477 | 4242 | 0.235716 | okx_context_ok | OKX public context does not obviously block a small repeat |
| POL | POL-USDT-SWAP | sector_perp_context;on_chain_flow | 2.7572 | 0.062509 | 43288780 | 1.2624 | 12958 | 0.077175 | okx_context_ok | OKX public context does not obviously block a small repeat |
| HYPE | HYPE-USDT-SWAP | on_chain_flow | 2.6935 | -0.088896 | 8012111 | 1.6291 | 222404 | 0.004496 | okx_context_ok | OKX public context does not obviously block a small repeat |
| WLD | WLD-USDT-SWAP | l2_imbalance | 2.3679 | -1.913127 | 894573882 | 2.1292 | 39080 | 0.025588 | okx_context_ok | OKX public context does not obviously block a small repeat |
| PEPE | PEPE-USDT-SWAP | liquidation | 2.1830 | 0.408116 | 44996059000000 | 3.5939 | 143513 | 0.006968 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BNB | BNB-USDT-SWAP | l2_imbalance;on_chain_flow | 4.8127 | -0.327998 | 145815 | 1.6746 | 316781 | 0.003157 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| BTC | BTC-USDT-SWAP | on_chain_flow | 3.0455 | 0.241883 | 136107 | 0.0158 | 245603 | 0.004072 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| CHIP | CHIP-USDT-SWAP | exchange_catalyst | 3.9957 | -0.114816 | 152251700 | 3.1611 | 3172 | 0.315269 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| MON | MON-USDT-SWAP | on_chain_flow | 3.8131 | -1.553936 | 397633320 | 4.6030 | 3302 | 0.302819 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STX | STX-USDT-SWAP | on_chain_flow | 3.4022 | 0.757379 | 4457116 | 5.3605 | 1399 | 0.714769 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| BERA | BERA-USDT-SWAP | on_chain_flow | 3.2612 | -0.427888 | 14014043 | 4.1314 | 3942 | 0.253666 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| OP | OP-USDT-SWAP | on_chain_flow | 2.9303 | 0.876000 | 132538000 | 10.3681 | 167 | 6.000402 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |

## Interpretation

OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only means the public venue context does not obviously block a small repeat observation; account fees, fill quality, and operational constraints are still unchecked.
