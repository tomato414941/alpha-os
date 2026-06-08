# Current Follow-Up OKX Execution Context

This joins follow-up candidates to current OKX USDT swap ticker, funding, spread, and public book depth. It is a rough venue context, not a fill model.

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | NEAR-USDT-SWAP | exchange_catalyst;on_chain_flow | 8.3498 | 0.876000 | 69912351 | 4.5610 | 23524 | 0.042510 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SOL | SOL-USDT-SWAP | exchange_catalyst;on_chain_flow | 5.2019 | 0.045470 | 11530187 | 1.4796 | 1050361 | 0.000952 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MEGA | MEGA-USDT-SWAP | exchange_catalyst;on_chain_flow | 4.4940 | -0.526928 | 195058140 | 1.9591 | 4904 | 0.203923 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ETH | ETH-USDT-SWAP | on_chain_flow | 3.9347 | -1.095513 | 5324182 | 0.0593 | 861837 | 0.001160 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SEI | SEI-USDT-SWAP | on_chain_flow | 3.8994 | -0.742966 | 59525600 | 2.0050 | 8187 | 0.122145 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ARB | ARB-USDT-SWAP | on_chain_flow | 3.7070 | -0.216386 | 141252547 | 1.2000 | 22508 | 0.044429 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SUI | SUI-USDT-SWAP | on_chain_flow | 3.6307 | 0.398045 | 107955058 | 1.3154 | 122022 | 0.008195 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ADA | ADA-USDT-SWAP | on_chain_flow | 3.4842 | -0.504460 | 415772390 | 5.8224 | 76249 | 0.013115 | okx_context_ok | OKX public context does not obviously block a small repeat |
| APT | APT-USDT-SWAP | on_chain_flow | 3.3984 | 0.505842 | 18746569 | 1.4657 | 18444 | 0.054219 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BERA | BERA-USDT-SWAP | on_chain_flow | 3.2612 | 0.403706 | 34571881 | 3.7168 | 4383 | 0.228143 | okx_context_ok | OKX public context does not obviously block a small repeat |
| POL | POL-USDT-SWAP | sector_perp_context;on_chain_flow | 2.7572 | -0.115243 | 63424160 | 1.2720 | 15414 | 0.064877 | okx_context_ok | OKX public context does not obviously block a small repeat |
| HYPE | HYPE-USDT-SWAP | on_chain_flow | 2.6935 | 0.822681 | 9616445 | 1.5720 | 239960 | 0.004167 | okx_context_ok | OKX public context does not obviously block a small repeat |
| WLD | WLD-USDT-SWAP | l2_imbalance | 2.3679 | 0.876000 | 944705614 | 1.8381 | 26494 | 0.037744 | okx_context_ok | OKX public context does not obviously block a small repeat |
| PEPE | PEPE-USDT-SWAP | liquidation | 2.1830 | -0.821379 | 42257007000000 | 3.5180 | 212530 | 0.004705 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BNB | BNB-USDT-SWAP | l2_imbalance;on_chain_flow | 4.8127 | 0.499430 | 130672 | 1.6468 | 298855 | 0.003346 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| BTC | BTC-USDT-SWAP | on_chain_flow | 3.0455 | -0.107184 | 147544 | 0.0158 | 187161 | 0.005343 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| CHIP | CHIP-USDT-SWAP | exchange_catalyst | 3.9957 | -1.115198 | 179084900 | 2.9669 | 3895 | 0.256724 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| MON | MON-USDT-SWAP | on_chain_flow | 3.8131 | -0.696340 | 350311230 | 4.6157 | 3955 | 0.252842 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STX | STX-USDT-SWAP | on_chain_flow | 3.4022 | -0.416227 | 4471639 | 5.3177 | 1119 | 0.893526 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STRK | STRK-USDT-SWAP | on_chain_flow | 3.3542 | 0.438000 | 116145485 | 2.8625 | 3722 | 0.268654 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| OP | OP-USDT-SWAP | on_chain_flow | 2.9303 | 0.876000 | 133341001 | 10.2512 | 4905 | 0.203878 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |

## Interpretation

OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only means the public venue context does not obviously block a small repeat observation; account fees, fill quality, and operational constraints are still unchecked.
