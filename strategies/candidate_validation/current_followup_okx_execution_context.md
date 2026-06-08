# Current Follow-Up OKX Execution Context

This joins follow-up candidates to current OKX USDT swap ticker, funding, spread, and public book depth. It is a rough venue context, not a fill model.

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | NEAR-USDT-SWAP | exchange_catalyst;on_chain_flow | 8.3498 | 0.876000 | 69441101 | 4.5589 | 22164 | 0.045118 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SOL | SOL-USDT-SWAP | exchange_catalyst;on_chain_flow | 5.2019 | 0.033174 | 11494424 | 1.4825 | 1064317 | 0.000940 | okx_context_ok | OKX public context does not obviously block a small repeat |
| CHIP | CHIP-USDT-SWAP | exchange_catalyst | 3.9957 | -1.487648 | 180572600 | 2.9757 | 4650 | 0.215034 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ETH | ETH-USDT-SWAP | on_chain_flow | 3.9347 | -1.180883 | 5296343 | 0.0592 | 1050376 | 0.000952 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SEI | SEI-USDT-SWAP | on_chain_flow | 3.8994 | -0.715616 | 58643500 | 2.0046 | 5843 | 0.171153 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MON | MON-USDT-SWAP | on_chain_flow | 3.8131 | -0.479472 | 348929690 | 4.6115 | 4394 | 0.227591 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ARB | ARB-USDT-SWAP | on_chain_flow | 3.7070 | -0.253407 | 140836738 | 1.1983 | 16484 | 0.060666 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SUI | SUI-USDT-SWAP | on_chain_flow | 3.6307 | 0.487273 | 106966262 | 1.3141 | 81064 | 0.012336 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ADA | ADA-USDT-SWAP | on_chain_flow | 3.4842 | -0.468886 | 413666830 | 5.8190 | 80809 | 0.012375 | okx_context_ok | OKX public context does not obviously block a small repeat |
| APT | APT-USDT-SWAP | on_chain_flow | 3.3984 | 0.393874 | 18635468 | 1.4645 | 21451 | 0.046618 | okx_context_ok | OKX public context does not obviously block a small repeat |
| STRK | STRK-USDT-SWAP | on_chain_flow | 3.3542 | 0.438000 | 114193474 | 2.8641 | 4063 | 0.246153 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BERA | BERA-USDT-SWAP | on_chain_flow | 3.2612 | 0.438000 | 34556346 | 3.7182 | 4033 | 0.247943 | okx_context_ok | OKX public context does not obviously block a small repeat |
| POL | POL-USDT-SWAP | sector_perp_context;on_chain_flow | 2.7572 | -0.105178 | 62832270 | 1.2722 | 12882 | 0.077629 | okx_context_ok | OKX public context does not obviously block a small repeat |
| HYPE | HYPE-USDT-SWAP | on_chain_flow | 2.6935 | 0.839387 | 9554408 | 1.5759 | 289520 | 0.003454 | okx_context_ok | OKX public context does not obviously block a small repeat |
| WLD | WLD-USDT-SWAP | l2_imbalance | 2.3679 | 0.876000 | 952228128 | 1.8230 | 45926 | 0.021774 | okx_context_ok | OKX public context does not obviously block a small repeat |
| PEPE | PEPE-USDT-SWAP | liquidation | 2.1830 | -0.759987 | 41899303000000 | 3.5180 | 88149 | 0.011344 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BNB | BNB-USDT-SWAP | l2_imbalance;on_chain_flow | 4.8127 | 0.510418 | 130166 | 1.6446 | 283943 | 0.003522 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| BTC | BTC-USDT-SWAP | on_chain_flow | 3.0455 | -0.075631 | 146123 | 0.0158 | 595211 | 0.001680 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| MEGA | MEGA-USDT-SWAP | exchange_catalyst;on_chain_flow | 4.4940 | -0.568883 | 192460730 | 1.9722 | 3836 | 0.260707 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STX | STX-USDT-SWAP | on_chain_flow | 3.4022 | -0.311121 | 4404353 | 5.3177 | 1777 | 0.562590 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| OP | OP-USDT-SWAP | on_chain_flow | 2.9303 | 0.876000 | 131066422 | 10.2617 | 1036 | 0.965491 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |

## Interpretation

OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only means the public venue context does not obviously block a small repeat observation; account fees, fill quality, and operational constraints are still unchecked.
