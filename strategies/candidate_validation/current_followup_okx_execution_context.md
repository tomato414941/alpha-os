# Current Follow-Up OKX Execution Context

This joins follow-up candidates to current OKX USDT swap ticker, funding, spread, and public book depth. It is a rough venue context, not a fill model.

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | NEAR-USDT-SWAP | exchange_catalyst;on_chain_flow | 8.3498 | 0.415541 | 73390475 | 4.7026 | 37029 | 0.027006 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SOL | SOL-USDT-SWAP | exchange_catalyst;on_chain_flow | 5.2019 | -1.639443 | 11993736 | 1.5178 | 935776 | 0.001069 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MEGA | MEGA-USDT-SWAP | exchange_catalyst;on_chain_flow | 4.4940 | -1.642577 | 213039960 | 1.9730 | 6355 | 0.157347 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ETH | ETH-USDT-SWAP | on_chain_flow | 3.9347 | -0.512831 | 5320270 | 0.0600 | 1604012 | 0.000623 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SEI | SEI-USDT-SWAP | on_chain_flow | 3.8994 | -0.304934 | 69257450 | 2.0151 | 7035 | 0.142152 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MON | MON-USDT-SWAP | on_chain_flow | 3.8131 | -1.928381 | 413213030 | 4.5966 | 4716 | 0.212066 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ARB | ARB-USDT-SWAP | on_chain_flow | 3.7070 | 0.657307 | 137941021 | 1.2194 | 19347 | 0.051687 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SUI | SUI-USDT-SWAP | on_chain_flow | 3.6307 | -0.350472 | 116424725 | 1.3216 | 82000 | 0.012195 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ADA | ADA-USDT-SWAP | on_chain_flow | 3.4842 | -0.571953 | 428114360 | 6.1331 | 81411 | 0.012283 | okx_context_ok | OKX public context does not obviously block a small repeat |
| APT | APT-USDT-SWAP | on_chain_flow | 3.3984 | -0.247942 | 19295737 | 1.5039 | 13909 | 0.071894 | okx_context_ok | OKX public context does not obviously block a small repeat |
| STRK | STRK-USDT-SWAP | on_chain_flow | 3.3542 | 0.438000 | 111237619 | 2.9669 | 4365 | 0.229081 | okx_context_ok | OKX public context does not obviously block a small repeat |
| POL | POL-USDT-SWAP | sector_perp_context;on_chain_flow | 2.7572 | -0.169912 | 46004580 | 1.2638 | 10385 | 0.096297 | okx_context_ok | OKX public context does not obviously block a small repeat |
| HYPE | HYPE-USDT-SWAP | on_chain_flow | 2.6935 | -0.208439 | 7871539 | 1.5867 | 205338 | 0.004870 | okx_context_ok | OKX public context does not obviously block a small repeat |
| WLD | WLD-USDT-SWAP | l2_imbalance | 2.3679 | -1.561889 | 940206789 | 2.1130 | 23798 | 0.042021 | okx_context_ok | OKX public context does not obviously block a small repeat |
| PEPE | PEPE-USDT-SWAP | liquidation | 2.1830 | -0.084093 | 48625284000000 | 3.5836 | 132993 | 0.007519 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BNB | BNB-USDT-SWAP | l2_imbalance;on_chain_flow | 4.8127 | -0.289413 | 153584 | 1.6825 | 414100 | 0.002415 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| BTC | BTC-USDT-SWAP | on_chain_flow | 3.0455 | 0.361081 | 145575 | 0.0158 | 417485 | 0.002395 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| CHIP | CHIP-USDT-SWAP | exchange_catalyst | 3.9957 | 0.401165 | 148911400 | 3.2378 | 3641 | 0.274673 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STX | STX-USDT-SWAP | on_chain_flow | 3.4022 | -0.670492 | 4487223 | 5.3548 | 855 | 1.168996 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| BERA | BERA-USDT-SWAP | on_chain_flow | 3.2612 | 0.065721 | 14004414 | 4.1623 | 1025 | 0.975274 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| OP | OP-USDT-SWAP | on_chain_flow | 2.9303 | 0.876000 | 137073816 | 10.3573 | 8143 | 0.122801 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |

## Interpretation

OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only means the public venue context does not obviously block a small repeat observation; account fees, fill quality, and operational constraints are still unchecked.
