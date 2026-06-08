# Current Follow-Up OKX Execution Context

This joins follow-up candidates to current OKX USDT swap ticker, funding, spread, and public book depth. It is a rough venue context, not a fill model.

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | NEAR-USDT-SWAP | exchange_catalyst;on_chain_flow | 8.3498 | 0.876000 | 69478828 | 4.5693 | 33070 | 0.030239 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SOL | SOL-USDT-SWAP | exchange_catalyst;on_chain_flow | 5.2019 | 0.045414 | 11505116 | 1.4827 | 956729 | 0.001045 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MEGA | MEGA-USDT-SWAP | exchange_catalyst;on_chain_flow | 4.4940 | -0.580964 | 192594540 | 1.9753 | 4780 | 0.209207 | okx_context_ok | OKX public context does not obviously block a small repeat |
| CHIP | CHIP-USDT-SWAP | exchange_catalyst | 3.9957 | -1.471236 | 180348500 | 2.9740 | 4338 | 0.230503 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ETH | ETH-USDT-SWAP | on_chain_flow | 3.9347 | -1.176093 | 5307881 | 0.0593 | 1065801 | 0.000938 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SEI | SEI-USDT-SWAP | on_chain_flow | 3.8994 | -0.720302 | 58938780 | 2.0074 | 6315 | 0.158341 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MON | MON-USDT-SWAP | on_chain_flow | 3.8131 | -0.493782 | 349127030 | 4.6157 | 4434 | 0.225531 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ARB | ARB-USDT-SWAP | on_chain_flow | 3.7070 | -0.241416 | 140804387 | 1.1997 | 22196 | 0.045053 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SUI | SUI-USDT-SWAP | on_chain_flow | 3.6307 | 0.472668 | 107092201 | 1.3162 | 92654 | 0.010793 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ADA | ADA-USDT-SWAP | on_chain_flow | 3.4842 | -0.465576 | 414165740 | 5.8326 | 68946 | 0.014504 | okx_context_ok | OKX public context does not obviously block a small repeat |
| APT | APT-USDT-SWAP | on_chain_flow | 3.3984 | 0.411315 | 18637373 | 1.4664 | 24660 | 0.040552 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BERA | BERA-USDT-SWAP | on_chain_flow | 3.2612 | 0.438000 | 34559695 | 3.7265 | 4106 | 0.243537 | okx_context_ok | OKX public context does not obviously block a small repeat |
| POL | POL-USDT-SWAP | sector_perp_context;on_chain_flow | 2.7572 | -0.107990 | 62608000 | 1.2732 | 13882 | 0.072036 | okx_context_ok | OKX public context does not obviously block a small repeat |
| HYPE | HYPE-USDT-SWAP | on_chain_flow | 2.6935 | 0.836723 | 9550421 | 1.5757 | 233761 | 0.004278 | okx_context_ok | OKX public context does not obviously block a small repeat |
| WLD | WLD-USDT-SWAP | l2_imbalance | 2.3679 | 0.876000 | 951669457 | 1.8065 | 43843 | 0.022808 | okx_context_ok | OKX public context does not obviously block a small repeat |
| PEPE | PEPE-USDT-SWAP | liquidation | 2.1830 | -0.805844 | 41841882000000 | 3.5217 | 145163 | 0.006889 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BNB | BNB-USDT-SWAP | l2_imbalance;on_chain_flow | 4.8127 | 0.511397 | 130285 | 1.6451 | 317098 | 0.003154 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| BTC | BTC-USDT-SWAP | on_chain_flow | 3.0455 | -0.079318 | 146395 | 0.0158 | 664803 | 0.001504 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| STX | STX-USDT-SWAP | on_chain_flow | 3.4022 | -0.307015 | 4393464 | 5.3262 | 1323 | 0.755776 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STRK | STRK-USDT-SWAP | on_chain_flow | 3.3542 | 0.438000 | 114238457 | 2.8715 | 3759 | 0.266049 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| OP | OP-USDT-SWAP | on_chain_flow | 2.9303 | 0.876000 | 130765975 | 10.2617 | 12416 | 0.080539 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |

## Interpretation

OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only means the public venue context does not obviously block a small repeat observation; account fees, fill quality, and operational constraints are still unchecked.
