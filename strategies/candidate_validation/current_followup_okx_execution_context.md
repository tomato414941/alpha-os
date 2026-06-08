# Current Follow-Up OKX Execution Context

This joins follow-up candidates to current OKX USDT swap ticker, funding, spread, and public book depth. It is a rough venue context, not a fill model.

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | NEAR-USDT-SWAP | exchange_catalyst;on_chain_flow | 8.3498 | 0.304239 | 69080917 | 4.7653 | 15209 | 0.065751 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SOL | SOL-USDT-SWAP | exchange_catalyst;on_chain_flow | 5.2019 | -1.882832 | 12297031 | 1.5266 | 893460 | 0.001119 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MEGA | MEGA-USDT-SWAP | exchange_catalyst;on_chain_flow | 4.4940 | -0.751603 | 195176520 | 2.0090 | 5398 | 0.185268 | okx_context_ok | OKX public context does not obviously block a small repeat |
| CHIP | CHIP-USDT-SWAP | exchange_catalyst | 3.9957 | -1.136071 | 153008700 | 3.2642 | 6491 | 0.154070 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ETH | ETH-USDT-SWAP | on_chain_flow | 3.9347 | -0.464208 | 5452748 | 0.0601 | 1294241 | 0.000773 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SEI | SEI-USDT-SWAP | on_chain_flow | 3.8994 | -0.874780 | 73744640 | 2.0406 | 7852 | 0.127359 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ARB | ARB-USDT-SWAP | on_chain_flow | 3.7070 | -0.067995 | 169159765 | 1.2250 | 14952 | 0.066881 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SUI | SUI-USDT-SWAP | on_chain_flow | 3.6307 | 0.010479 | 116872885 | 1.3460 | 94796 | 0.010549 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ADA | ADA-USDT-SWAP | on_chain_flow | 3.4842 | -0.392665 | 439792060 | 6.2054 | 90793 | 0.011014 | okx_context_ok | OKX public context does not obviously block a small repeat |
| APT | APT-USDT-SWAP | on_chain_flow | 3.3984 | -0.647906 | 19845028 | 1.5109 | 14249 | 0.070182 | okx_context_ok | OKX public context does not obviously block a small repeat |
| POL | POL-USDT-SWAP | sector_perp_context;on_chain_flow | 2.7572 | 0.438000 | 49104160 | 1.2649 | 9090 | 0.110007 | okx_context_ok | OKX public context does not obviously block a small repeat |
| HYPE | HYPE-USDT-SWAP | on_chain_flow | 2.6935 | -0.943055 | 7554147 | 1.6001 | 311174 | 0.003214 | okx_context_ok | OKX public context does not obviously block a small repeat |
| WLD | WLD-USDT-SWAP | l2_imbalance | 2.3679 | -1.741737 | 958209210 | 2.1081 | 33101 | 0.030210 | okx_context_ok | OKX public context does not obviously block a small repeat |
| PEPE | PEPE-USDT-SWAP | liquidation | 2.1830 | -0.295779 | 49813577000000 | 3.6095 | 143638 | 0.006962 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BNB | BNB-USDT-SWAP | l2_imbalance;on_chain_flow | 4.8127 | 0.063908 | 158142 | 1.6791 | 390521 | 0.002561 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| BTC | BTC-USDT-SWAP | on_chain_flow | 3.0455 | 0.523588 | 148834 | 0.0159 | 380594 | 0.002627 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| MON | MON-USDT-SWAP | on_chain_flow | 3.8131 | -2.350667 | 431849130 | 4.5988 | 2099 | 0.476333 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STX | STX-USDT-SWAP | on_chain_flow | 3.4022 | -1.467537 | 4753660 | 10.7469 | 518 | 1.931859 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |
| STRK | STRK-USDT-SWAP | on_chain_flow | 3.3542 | 0.438000 | 118404376 | 2.9766 | 3703 | 0.270057 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| BERA | BERA-USDT-SWAP | on_chain_flow | 3.2612 | 0.438000 | 13612148 | 4.1955 | 1119 | 0.893299 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| OP | OP-USDT-SWAP | on_chain_flow | 2.9303 | 0.811688 | 141282550 | 10.4439 | 5696 | 0.175551 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |

## Interpretation

OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only means the public venue context does not obviously block a small repeat observation; account fees, fill quality, and operational constraints are still unchecked.
