# Current Follow-Up OKX Execution Context

This joins follow-up candidates to current OKX USDT swap ticker, funding, spread, and public book depth. It is a rough venue context, not a fill model.

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | NEAR-USDT-SWAP | exchange_catalyst;on_chain_flow | 8.3498 | 0.364553 | 73198709 | 4.6740 | 30445 | 0.032846 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SOL | SOL-USDT-SWAP | exchange_catalyst;on_chain_flow | 5.2019 | -1.644513 | 11990573 | 1.5132 | 1059133 | 0.000944 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MEGA | MEGA-USDT-SWAP | exchange_catalyst;on_chain_flow | 4.4940 | -1.668077 | 211667340 | 1.9730 | 6904 | 0.144843 | okx_context_ok | OKX public context does not obviously block a small repeat |
| CHIP | CHIP-USDT-SWAP | exchange_catalyst | 3.9957 | 0.438000 | 148362500 | 3.2295 | 4725 | 0.211623 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ETH | ETH-USDT-SWAP | on_chain_flow | 3.9347 | -0.511830 | 5311086 | 0.0598 | 1236968 | 0.000808 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SEI | SEI-USDT-SWAP | on_chain_flow | 3.8994 | -0.314718 | 69375980 | 2.0131 | 9444 | 0.105889 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ARB | ARB-USDT-SWAP | on_chain_flow | 3.7070 | 0.669332 | 138196684 | 1.2160 | 14384 | 0.069523 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SUI | SUI-USDT-SWAP | on_chain_flow | 3.6307 | -0.316536 | 116340968 | 1.3162 | 87061 | 0.011486 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ADA | ADA-USDT-SWAP | on_chain_flow | 3.4842 | -0.561195 | 427592680 | 6.1143 | 77541 | 0.012896 | okx_context_ok | OKX public context does not obviously block a small repeat |
| APT | APT-USDT-SWAP | on_chain_flow | 3.3984 | -0.230309 | 19513012 | 1.4996 | 9568 | 0.104520 | okx_context_ok | OKX public context does not obviously block a small repeat |
| POL | POL-USDT-SWAP | sector_perp_context;on_chain_flow | 2.7572 | -0.135955 | 46105850 | 1.2600 | 8990 | 0.111229 | okx_context_ok | OKX public context does not obviously block a small repeat |
| HYPE | HYPE-USDT-SWAP | on_chain_flow | 2.6935 | -0.219191 | 7862463 | 1.5877 | 261007 | 0.003831 | okx_context_ok | OKX public context does not obviously block a small repeat |
| WLD | WLD-USDT-SWAP | l2_imbalance | 2.3679 | -1.509275 | 940944290 | 2.1166 | 35226 | 0.028388 | okx_context_ok | OKX public context does not obviously block a small repeat |
| PEPE | PEPE-USDT-SWAP | liquidation | 2.1830 | -0.115836 | 48639090000000 | 3.5759 | 132673 | 0.007537 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BNB | BNB-USDT-SWAP | l2_imbalance;on_chain_flow | 4.8127 | -0.280720 | 153466 | 1.6780 | 407217 | 0.002456 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| BTC | BTC-USDT-SWAP | on_chain_flow | 3.0455 | 0.373931 | 145727 | 0.0158 | 330518 | 0.003026 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| MON | MON-USDT-SWAP | on_chain_flow | 3.8131 | -1.900566 | 414298120 | 4.5777 | 3994 | 0.250374 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STX | STX-USDT-SWAP | on_chain_flow | 3.4022 | -0.734737 | 4508382 | 10.6667 | 199 | 5.014572 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |
| STRK | STRK-USDT-SWAP | on_chain_flow | 3.3542 | 0.438000 | 111533731 | 2.9564 | 3048 | 0.328038 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| BERA | BERA-USDT-SWAP | on_chain_flow | 3.2612 | 0.086665 | 13994229 | 4.1434 | 2143 | 0.466601 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| OP | OP-USDT-SWAP | on_chain_flow | 2.9303 | 0.876000 | 137300485 | 10.3146 | 771 | 1.296438 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |

## Interpretation

OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only means the public venue context does not obviously block a small repeat observation; account fees, fill quality, and operational constraints are still unchecked.
