# Current Follow-Up OKX Execution Context

This joins follow-up candidates to current OKX USDT swap ticker, funding, spread, and public book depth. It is a rough venue context, not a fill model.

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | NEAR-USDT-SWAP | exchange_catalyst;on_chain_flow | 8.3498 | 0.876000 | 74497853 | 4.6328 | 33244 | 0.030081 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SOL | SOL-USDT-SWAP | exchange_catalyst;on_chain_flow | 5.2019 | -1.411749 | 11099876 | 1.5109 | 1348209 | 0.000742 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MEGA | MEGA-USDT-SWAP | exchange_catalyst;on_chain_flow | 4.4940 | -4.740505 | 247574020 | 1.9800 | 4635 | 0.215755 | okx_context_ok | OKX public context does not obviously block a small repeat |
| CHIP | CHIP-USDT-SWAP | exchange_catalyst | 3.9957 | -0.502731 | 154897700 | 3.1691 | 5191 | 0.192641 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ETH | ETH-USDT-SWAP | on_chain_flow | 3.9347 | -0.425120 | 4915295 | 0.0600 | 1211266 | 0.000826 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SEI | SEI-USDT-SWAP | on_chain_flow | 3.8994 | -0.392550 | 66936020 | 1.9982 | 9323 | 0.107258 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ARB | ARB-USDT-SWAP | on_chain_flow | 3.7070 | 0.847812 | 134773468 | 1.2181 | 20359 | 0.049118 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SUI | SUI-USDT-SWAP | on_chain_flow | 3.6307 | -0.229143 | 112050616 | 1.3293 | 97509 | 0.010255 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ADA | ADA-USDT-SWAP | on_chain_flow | 3.4842 | 0.518475 | 410987050 | 6.0883 | 75958 | 0.013165 | okx_context_ok | OKX public context does not obviously block a small repeat |
| APT | APT-USDT-SWAP | on_chain_flow | 3.3984 | 0.481763 | 17712368 | 1.4985 | 19619 | 0.050971 | okx_context_ok | OKX public context does not obviously block a small repeat |
| POL | POL-USDT-SWAP | sector_perp_context;on_chain_flow | 2.7572 | 0.148171 | 44033740 | 1.2545 | 11084 | 0.090221 | okx_context_ok | OKX public context does not obviously block a small repeat |
| HYPE | HYPE-USDT-SWAP | on_chain_flow | 2.6935 | -0.161686 | 7954279 | 1.6280 | 220816 | 0.004529 | okx_context_ok | OKX public context does not obviously block a small repeat |
| WLD | WLD-USDT-SWAP | l2_imbalance | 2.3679 | -2.161129 | 903441173 | 2.1297 | 35466 | 0.028196 | okx_context_ok | OKX public context does not obviously block a small repeat |
| PEPE | PEPE-USDT-SWAP | liquidation | 2.1830 | 0.434123 | 45107535000000 | 3.5772 | 114124 | 0.008762 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BNB | BNB-USDT-SWAP | l2_imbalance;on_chain_flow | 4.8127 | -0.264557 | 145706 | 1.6699 | 348149 | 0.002872 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| BTC | BTC-USDT-SWAP | on_chain_flow | 3.0455 | 0.209190 | 136139 | 0.0158 | 294851 | 0.003392 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| MON | MON-USDT-SWAP | on_chain_flow | 3.8131 | -1.319908 | 397693030 | 4.5777 | 2999 | 0.333404 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STX | STX-USDT-SWAP | on_chain_flow | 3.4022 | 0.876000 | 4500143 | 5.3433 | 1562 | 0.640113 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STRK | STRK-USDT-SWAP | on_chain_flow | 3.3542 | 0.438000 | 106077170 | 2.9313 | 3511 | 0.284787 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| BERA | BERA-USDT-SWAP | on_chain_flow | 3.2612 | -0.487822 | 14543946 | 4.0527 | 2924 | 0.342010 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| OP | OP-USDT-SWAP | on_chain_flow | 2.9303 | 0.876000 | 132468873 | 10.2934 | 4123 | 0.242523 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |

## Interpretation

OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only means the public venue context does not obviously block a small repeat observation; account fees, fill quality, and operational constraints are still unchecked.
