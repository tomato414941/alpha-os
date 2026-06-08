# Current Follow-Up OKX Execution Context

This joins follow-up candidates to current OKX USDT swap ticker, funding, spread, and public book depth. It is a rough venue context, not a fill model.

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | NEAR-USDT-SWAP | exchange_catalyst;on_chain_flow | 8.3498 | 0.876000 | 66985783 | 4.6959 | 29999 | 0.033335 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SOL | SOL-USDT-SWAP | exchange_catalyst;on_chain_flow | 5.2019 | -0.708211 | 10298007 | 1.4998 | 973407 | 0.001027 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MEGA | MEGA-USDT-SWAP | exchange_catalyst;on_chain_flow | 4.4940 | -1.273280 | 183396300 | 1.9954 | 4568 | 0.218908 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ETH | ETH-USDT-SWAP | on_chain_flow | 3.9347 | -1.359817 | 4731124 | 0.0590 | 987396 | 0.001013 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SEI | SEI-USDT-SWAP | on_chain_flow | 3.8994 | -0.954299 | 52662330 | 2.0323 | 6922 | 0.144474 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ARB | ARB-USDT-SWAP | on_chain_flow | 3.7070 | 0.876000 | 121524398 | 1.2214 | 12684 | 0.078837 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SUI | SUI-USDT-SWAP | on_chain_flow | 3.6307 | 0.428423 | 86972632 | 1.3242 | 149906 | 0.006671 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ADA | ADA-USDT-SWAP | on_chain_flow | 3.4842 | -1.015159 | 356242860 | 5.8668 | 68542 | 0.014590 | okx_context_ok | OKX public context does not obviously block a small repeat |
| APT | APT-USDT-SWAP | on_chain_flow | 3.3984 | 0.595926 | 16195233 | 1.4909 | 17621 | 0.056750 | okx_context_ok | OKX public context does not obviously block a small repeat |
| POL | POL-USDT-SWAP | sector_perp_context;on_chain_flow | 2.7572 | -0.004367 | 57121740 | 1.2790 | 8601 | 0.116260 | okx_context_ok | OKX public context does not obviously block a small repeat |
| HYPE | HYPE-USDT-SWAP | on_chain_flow | 2.6935 | 0.151902 | 9313622 | 1.5832 | 135463 | 0.007382 | okx_context_ok | OKX public context does not obviously block a small repeat |
| WLD | WLD-USDT-SWAP | l2_imbalance | 2.3679 | 0.876000 | 1004522619 | 2.0229 | 38240 | 0.026151 | okx_context_ok | OKX public context does not obviously block a small repeat |
| PEPE | PEPE-USDT-SWAP | liquidation | 2.1830 | -0.198446 | 32948572000000 | 3.5581 | 163518 | 0.006116 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BNB | BNB-USDT-SWAP | l2_imbalance;on_chain_flow | 4.8127 | 0.370041 | 102499 | 1.6621 | 269413 | 0.003712 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| BTC | BTC-USDT-SWAP | on_chain_flow | 3.0455 | -0.149683 | 117161 | 0.0158 | 432654 | 0.002311 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| CHIP | CHIP-USDT-SWAP | exchange_catalyst | 3.9957 | -2.262628 | 188493500 | 2.9999 | 3745 | 0.267049 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| MON | MON-USDT-SWAP | on_chain_flow | 3.8131 | -0.681587 | 320560620 | 4.7973 | 3855 | 0.259419 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STX | STX-USDT-SWAP | on_chain_flow | 3.4022 | 0.876000 | 3768824 | 5.4333 | 374 | 2.671951 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STRK | STRK-USDT-SWAP | on_chain_flow | 3.3542 | 0.438000 | 106349586 | 2.9304 | 3195 | 0.312945 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| BERA | BERA-USDT-SWAP | on_chain_flow | 3.2612 | 0.438000 | 33523134 | 3.8873 | 3344 | 0.299064 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| OP | OP-USDT-SWAP | on_chain_flow | 2.9303 | 0.876000 | 94276051 | 10.4221 | 3530 | 0.283280 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |

## Interpretation

OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only means the public venue context does not obviously block a small repeat observation; account fees, fill quality, and operational constraints are still unchecked.
