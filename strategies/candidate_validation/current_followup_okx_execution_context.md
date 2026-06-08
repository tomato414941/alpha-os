# Current Follow-Up OKX Execution Context

This joins follow-up candidates to current OKX USDT swap ticker, funding, spread, and public book depth. It is a rough venue context, not a fill model.

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | NEAR-USDT-SWAP | exchange_catalyst;on_chain_flow | 8.3498 | 0.705984 | 74094540 | 4.7181 | 33194 | 0.030126 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SOL | SOL-USDT-SWAP | exchange_catalyst;on_chain_flow | 5.2019 | -1.558002 | 11593126 | 1.5280 | 1166762 | 0.000857 | okx_context_ok | OKX public context does not obviously block a small repeat |
| CHIP | CHIP-USDT-SWAP | exchange_catalyst | 3.9957 | 0.432291 | 151777700 | 3.1721 | 5221 | 0.191552 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ETH | ETH-USDT-SWAP | on_chain_flow | 3.9347 | -0.466361 | 5132279 | 0.0603 | 514216 | 0.001945 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SEI | SEI-USDT-SWAP | on_chain_flow | 3.8994 | -0.551779 | 68174330 | 2.0398 | 10576 | 0.094552 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MON | MON-USDT-SWAP | on_chain_flow | 3.8131 | -1.983009 | 406595770 | 4.6307 | 4364 | 0.229152 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ARB | ARB-USDT-SWAP | on_chain_flow | 3.7070 | 0.746911 | 135839462 | 1.2304 | 16547 | 0.060435 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SUI | SUI-USDT-SWAP | on_chain_flow | 3.6307 | -0.268523 | 114950738 | 1.3393 | 49843 | 0.020063 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ADA | ADA-USDT-SWAP | on_chain_flow | 3.4842 | -0.127691 | 425461530 | 6.1595 | 56875 | 0.017583 | okx_context_ok | OKX public context does not obviously block a small repeat |
| APT | APT-USDT-SWAP | on_chain_flow | 3.3984 | 0.200942 | 17923872 | 1.5169 | 10979 | 0.091084 | okx_context_ok | OKX public context does not obviously block a small repeat |
| POL | POL-USDT-SWAP | sector_perp_context;on_chain_flow | 2.7572 | -0.187577 | 44693470 | 1.2696 | 7525 | 0.132895 | okx_context_ok | OKX public context does not obviously block a small repeat |
| HYPE | HYPE-USDT-SWAP | on_chain_flow | 2.6935 | -0.082650 | 8071014 | 1.6349 | 187888 | 0.005322 | okx_context_ok | OKX public context does not obviously block a small repeat |
| WLD | WLD-USDT-SWAP | l2_imbalance | 2.3679 | -1.711663 | 921988944 | 2.1297 | 32550 | 0.030722 | okx_context_ok | OKX public context does not obviously block a small repeat |
| PEPE | PEPE-USDT-SWAP | liquidation | 2.1830 | 0.107076 | 47417614000000 | 3.6134 | 98098 | 0.010194 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BNB | BNB-USDT-SWAP | l2_imbalance;on_chain_flow | 4.8127 | -0.350968 | 147578 | 1.6856 | 330144 | 0.003029 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| BTC | BTC-USDT-SWAP | on_chain_flow | 3.0455 | 0.287623 | 136701 | 0.0159 | 463895 | 0.002156 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| MEGA | MEGA-USDT-SWAP | exchange_catalyst;on_chain_flow | 4.4940 | -3.942057 | 236787100 | 1.9568 | 3741 | 0.267288 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STX | STX-USDT-SWAP | on_chain_flow | 3.4022 | 0.017487 | 4391636 | 5.3807 | 1489 | 0.671738 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STRK | STRK-USDT-SWAP | on_chain_flow | 3.3542 | 0.438000 | 108339093 | 2.9652 | 2982 | 0.335303 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| BERA | BERA-USDT-SWAP | on_chain_flow | 3.2612 | -0.207192 | 14046630 | 4.1762 | 1967 | 0.508277 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| OP | OP-USDT-SWAP | on_chain_flow | 2.9303 | 0.876000 | 135277481 | 10.4221 | 3659 | 0.273294 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |

## Interpretation

OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only means the public venue context does not obviously block a small repeat observation; account fees, fill quality, and operational constraints are still unchecked.
