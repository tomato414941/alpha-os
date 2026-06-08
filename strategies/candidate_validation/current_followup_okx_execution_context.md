# Current Follow-Up OKX Execution Context

This joins follow-up candidates to current OKX USDT swap ticker, funding, spread, and public book depth. It is a rough venue context, not a fill model.

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | NEAR-USDT-SWAP | exchange_catalyst;on_chain_flow | 8.3498 | 0.876000 | 74637988 | 4.5861 | 32997 | 0.030306 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SOL | SOL-USDT-SWAP | exchange_catalyst;on_chain_flow | 5.2019 | -1.429873 | 10982051 | 1.5095 | 1320534 | 0.000757 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MEGA | MEGA-USDT-SWAP | exchange_catalyst;on_chain_flow | 4.4940 | -4.439210 | 251749610 | 1.9942 | 5407 | 0.184952 | okx_context_ok | OKX public context does not obviously block a small repeat |
| CHIP | CHIP-USDT-SWAP | exchange_catalyst | 3.9957 | -1.095346 | 153966200 | 3.1781 | 4277 | 0.233832 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ETH | ETH-USDT-SWAP | on_chain_flow | 3.9347 | -0.457239 | 4908938 | 0.0599 | 1335705 | 0.000749 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SEI | SEI-USDT-SWAP | on_chain_flow | 3.8994 | -0.571595 | 66488730 | 2.0006 | 9931 | 0.100699 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ARB | ARB-USDT-SWAP | on_chain_flow | 3.7070 | 0.876000 | 135934644 | 1.2112 | 16996 | 0.058837 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SUI | SUI-USDT-SWAP | on_chain_flow | 3.6307 | -0.226486 | 111087542 | 1.3290 | 101154 | 0.009886 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ADA | ADA-USDT-SWAP | on_chain_flow | 3.4842 | 0.651652 | 411961360 | 6.0478 | 71358 | 0.014014 | okx_context_ok | OKX public context does not obviously block a small repeat |
| APT | APT-USDT-SWAP | on_chain_flow | 3.3984 | 0.465138 | 17754285 | 1.4900 | 15587 | 0.064154 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BERA | BERA-USDT-SWAP | on_chain_flow | 3.2612 | -0.525222 | 14947975 | 8.0192 | 5335 | 0.187456 | okx_context_ok | OKX public context does not obviously block a small repeat |
| POL | POL-USDT-SWAP | sector_perp_context;on_chain_flow | 2.7572 | 0.216076 | 43915420 | 2.5078 | 9286 | 0.107691 | okx_context_ok | OKX public context does not obviously block a small repeat |
| HYPE | HYPE-USDT-SWAP | on_chain_flow | 2.6935 | -0.272701 | 7846656 | 1.6296 | 216830 | 0.004612 | okx_context_ok | OKX public context does not obviously block a small repeat |
| WLD | WLD-USDT-SWAP | l2_imbalance | 2.3679 | -2.425718 | 893528161 | 2.1274 | 46334 | 0.021582 | okx_context_ok | OKX public context does not obviously block a small repeat |
| PEPE | PEPE-USDT-SWAP | liquidation | 2.1830 | 0.545512 | 45046162000000 | 3.5682 | 178519 | 0.005602 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BNB | BNB-USDT-SWAP | l2_imbalance;on_chain_flow | 4.8127 | -0.203855 | 144952 | 1.6693 | 368564 | 0.002713 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| BTC | BTC-USDT-SWAP | on_chain_flow | 3.0455 | 0.155614 | 135933 | 0.0158 | 310215 | 0.003224 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| MON | MON-USDT-SWAP | on_chain_flow | 3.8131 | -1.099628 | 395636570 | 4.5966 | 2225 | 0.449416 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STX | STX-USDT-SWAP | on_chain_flow | 3.4022 | 0.876000 | 4466690 | 5.3433 | 1500 | 0.666482 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STRK | STRK-USDT-SWAP | on_chain_flow | 3.3542 | 0.438000 | 105468672 | 2.8998 | 3130 | 0.319466 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| OP | OP-USDT-SWAP | on_chain_flow | 2.9303 | 0.876000 | 132121260 | 10.2407 | 502 | 1.990268 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |

## Interpretation

OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only means the public venue context does not obviously block a small repeat observation; account fees, fill quality, and operational constraints are still unchecked.
