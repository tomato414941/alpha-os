# Current Follow-Up OKX Execution Context

This joins follow-up candidates to current OKX USDT swap ticker, funding, spread, and public book depth. It is a rough venue context, not a fill model.

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | NEAR-USDT-SWAP | exchange_catalyst;on_chain_flow | 8.3498 | 0.876000 | 69359937 | 4.6072 | 32586 | 0.030688 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SOL | SOL-USDT-SWAP | exchange_catalyst;on_chain_flow | 5.2019 | -0.135602 | 11621780 | 1.4880 | 776332 | 0.001288 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MEGA | MEGA-USDT-SWAP | exchange_catalyst;on_chain_flow | 4.4940 | -0.810009 | 191233740 | 1.9769 | 4739 | 0.211031 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ETH | ETH-USDT-SWAP | on_chain_flow | 3.9347 | -1.349190 | 5246445 | 0.0593 | 708009 | 0.001412 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SEI | SEI-USDT-SWAP | on_chain_flow | 3.8994 | -1.125278 | 57397670 | 4.0064 | 6618 | 0.151098 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MON | MON-USDT-SWAP | on_chain_flow | 3.8131 | -0.739569 | 351555360 | 4.6762 | 5036 | 0.198560 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ARB | ARB-USDT-SWAP | on_chain_flow | 3.7070 | -0.006242 | 140864222 | 1.2066 | 17911 | 0.055832 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SUI | SUI-USDT-SWAP | on_chain_flow | 3.6307 | 0.663570 | 105494742 | 1.3174 | 136135 | 0.007346 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ADA | ADA-USDT-SWAP | on_chain_flow | 3.4842 | -0.389184 | 413103880 | 5.8668 | 53724 | 0.018614 | okx_context_ok | OKX public context does not obviously block a small repeat |
| APT | APT-USDT-SWAP | on_chain_flow | 3.3984 | 0.507460 | 18635757 | 1.4742 | 21560 | 0.046381 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BERA | BERA-USDT-SWAP | on_chain_flow | 3.2612 | 0.438000 | 34666592 | 3.7573 | 4386 | 0.228009 | okx_context_ok | OKX public context does not obviously block a small repeat |
| POL | POL-USDT-SWAP | sector_perp_context;on_chain_flow | 2.7572 | -0.132518 | 61644550 | 1.2646 | 11306 | 0.088449 | okx_context_ok | OKX public context does not obviously block a small repeat |
| HYPE | HYPE-USDT-SWAP | on_chain_flow | 2.6935 | 0.800010 | 9511600 | 1.5697 | 229548 | 0.004356 | okx_context_ok | OKX public context does not obviously block a small repeat |
| WLD | WLD-USDT-SWAP | l2_imbalance | 2.3679 | 0.876000 | 971768160 | 1.8418 | 44086 | 0.022683 | okx_context_ok | OKX public context does not obviously block a small repeat |
| PEPE | PEPE-USDT-SWAP | liquidation | 2.1830 | -0.284294 | 41123386000000 | 3.5392 | 153228 | 0.006526 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BNB | BNB-USDT-SWAP | l2_imbalance;on_chain_flow | 4.8127 | 0.575412 | 128173 | 1.6508 | 335118 | 0.002984 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| BTC | BTC-USDT-SWAP | on_chain_flow | 3.0455 | -0.019307 | 143016 | 0.0158 | 348815 | 0.002867 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| CHIP | CHIP-USDT-SWAP | exchange_catalyst | 3.9957 | -2.679730 | 187071400 | 2.9599 | 3407 | 0.293480 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STX | STX-USDT-SWAP | on_chain_flow | 3.4022 | 0.452492 | 4563401 | 5.3548 | 1456 | 0.686645 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STRK | STRK-USDT-SWAP | on_chain_flow | 3.3542 | 0.438000 | 113300620 | 2.8823 | 3618 | 0.276390 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| OP | OP-USDT-SWAP | on_chain_flow | 2.9303 | 0.876000 | 126627003 | 10.3146 | 9183 | 0.108897 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |

## Interpretation

OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only means the public venue context does not obviously block a small repeat observation; account fees, fill quality, and operational constraints are still unchecked.
