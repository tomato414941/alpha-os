# Current Follow-Up OKX Execution Context

This joins follow-up candidates to current OKX USDT swap ticker, funding, spread, and public book depth. It is a rough venue context, not a fill model.

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | NEAR-USDT-SWAP | exchange_catalyst;on_chain_flow | 8.3498 | 0.298995 | 68384539 | 4.9322 | 10890 | 0.091831 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SOL | SOL-USDT-SWAP | exchange_catalyst;on_chain_flow | 5.2019 | -1.874607 | 12282391 | 1.5360 | 967360 | 0.001034 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MEGA | MEGA-USDT-SWAP | exchange_catalyst;on_chain_flow | 4.4940 | -0.630294 | 187288990 | 2.0892 | 4609 | 0.216987 | okx_context_ok | OKX public context does not obviously block a small repeat |
| CHIP | CHIP-USDT-SWAP | exchange_catalyst | 3.9957 | -1.637993 | 153505800 | 3.3063 | 5286 | 0.189172 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ETH | ETH-USDT-SWAP | on_chain_flow | 3.9347 | -0.453727 | 5385180 | 0.0607 | 1098579 | 0.000910 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SEI | SEI-USDT-SWAP | on_chain_flow | 3.8994 | -1.031922 | 73841220 | 2.0629 | 5879 | 0.170109 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ARB | ARB-USDT-SWAP | on_chain_flow | 3.7070 | -0.363761 | 174864537 | 1.2343 | 18154 | 0.055086 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SUI | SUI-USDT-SWAP | on_chain_flow | 3.6307 | 0.075988 | 116346049 | 1.3670 | 69797 | 0.014327 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ADA | ADA-USDT-SWAP | on_chain_flow | 3.4842 | -0.355626 | 439249910 | 6.2402 | 98169 | 0.010187 | okx_context_ok | OKX public context does not obviously block a small repeat |
| APT | APT-USDT-SWAP | on_chain_flow | 3.3984 | -0.985826 | 19819219 | 1.5245 | 16534 | 0.060482 | okx_context_ok | OKX public context does not obviously block a small repeat |
| POL | POL-USDT-SWAP | sector_perp_context;on_chain_flow | 2.7572 | 0.438000 | 49430190 | 1.2727 | 9908 | 0.100931 | okx_context_ok | OKX public context does not obviously block a small repeat |
| HYPE | HYPE-USDT-SWAP | on_chain_flow | 2.6935 | -1.188548 | 7411586 | 1.6381 | 232013 | 0.004310 | okx_context_ok | OKX public context does not obviously block a small repeat |
| WLD | WLD-USDT-SWAP | l2_imbalance | 2.3679 | -1.687482 | 953714590 | 2.1261 | 45230 | 0.022109 | okx_context_ok | OKX public context does not obviously block a small repeat |
| PEPE | PEPE-USDT-SWAP | liquidation | 2.1830 | -0.463955 | 49706210000000 | 3.6370 | 142169 | 0.007034 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BNB | BNB-USDT-SWAP | l2_imbalance;on_chain_flow | 4.8127 | 0.106547 | 157572 | 1.6865 | 286454 | 0.003491 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| BTC | BTC-USDT-SWAP | on_chain_flow | 3.0455 | 0.512002 | 147289 | 0.0160 | 350810 | 0.002851 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| MON | MON-USDT-SWAP | on_chain_flow | 3.8131 | -2.575868 | 433310910 | 4.6631 | 3384 | 0.295469 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STX | STX-USDT-SWAP | on_chain_flow | 3.4022 | -2.196158 | 4805671 | 5.4274 | 1769 | 0.565342 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STRK | STRK-USDT-SWAP | on_chain_flow | 3.3542 | 0.406002 | 118405597 | 3.0044 | 2588 | 0.386414 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| BERA | BERA-USDT-SWAP | on_chain_flow | 3.2612 | 0.036067 | 13554120 | 4.2364 | 3463 | 0.288758 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| OP | OP-USDT-SWAP | on_chain_flow | 2.9303 | 0.641992 | 142452741 | 10.5541 | 4377 | 0.228445 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |

## Interpretation

OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only means the public venue context does not obviously block a small repeat observation; account fees, fill quality, and operational constraints are still unchecked.
