# Current Follow-Up OKX Execution Context

This joins follow-up candidates to current OKX USDT swap ticker, funding, spread, and public book depth. It is a rough venue context, not a fill model.

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | NEAR-USDT-SWAP | exchange_catalyst;on_chain_flow | 8.3498 | -0.190621 | 78310761 | 4.6740 | 42691 | 0.023424 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SOL | SOL-USDT-SWAP | exchange_catalyst;on_chain_flow | 5.2019 | -0.798606 | 11049121 | 1.4991 | 1314599 | 0.000761 | okx_context_ok | OKX public context does not obviously block a small repeat |
| MEGA | MEGA-USDT-SWAP | exchange_catalyst;on_chain_flow | 4.4940 | -3.086100 | 286394340 | 1.9934 | 6060 | 0.165025 | okx_context_ok | OKX public context does not obviously block a small repeat |
| CHIP | CHIP-USDT-SWAP | exchange_catalyst | 3.9957 | -1.350786 | 163123900 | 3.0698 | 5368 | 0.186279 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ETH | ETH-USDT-SWAP | on_chain_flow | 3.9347 | -0.631230 | 5299388 | 0.0593 | 994630 | 0.001005 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SEI | SEI-USDT-SWAP | on_chain_flow | 3.8994 | -0.704607 | 63590710 | 2.0002 | 8733 | 0.114512 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ARB | ARB-USDT-SWAP | on_chain_flow | 3.7070 | 0.158209 | 145254904 | 1.1987 | 13697 | 0.073007 | okx_context_ok | OKX public context does not obviously block a small repeat |
| SUI | SUI-USDT-SWAP | on_chain_flow | 3.6307 | 0.000569 | 108294265 | 1.3143 | 121906 | 0.008203 | okx_context_ok | OKX public context does not obviously block a small repeat |
| ADA | ADA-USDT-SWAP | on_chain_flow | 3.4842 | 0.876000 | 411389750 | 6.0259 | 69455 | 0.014398 | okx_context_ok | OKX public context does not obviously block a small repeat |
| APT | APT-USDT-SWAP | on_chain_flow | 3.3984 | 0.400026 | 17603473 | 1.4829 | 13916 | 0.071859 | okx_context_ok | OKX public context does not obviously block a small repeat |
| STRK | STRK-USDT-SWAP | on_chain_flow | 3.3542 | 0.067626 | 119481432 | 2.8535 | 4342 | 0.230295 | okx_context_ok | OKX public context does not obviously block a small repeat |
| POL | POL-USDT-SWAP | sector_perp_context;on_chain_flow | 2.7572 | 0.136287 | 48464570 | 1.2415 | 11804 | 0.084716 | okx_context_ok | OKX public context does not obviously block a small repeat |
| HYPE | HYPE-USDT-SWAP | on_chain_flow | 2.6935 | -0.495697 | 8747697 | 1.5474 | 168820 | 0.005923 | okx_context_ok | OKX public context does not obviously block a small repeat |
| WLD | WLD-USDT-SWAP | l2_imbalance | 2.3679 | -3.904483 | 898672107 | 2.0936 | 33293 | 0.030037 | okx_context_ok | OKX public context does not obviously block a small repeat |
| PEPE | PEPE-USDT-SWAP | liquidation | 2.1830 | -0.855830 | 44268156000000 | 3.5455 | 152794 | 0.006545 | okx_context_ok | OKX public context does not obviously block a small repeat |
| BNB | BNB-USDT-SWAP | l2_imbalance;on_chain_flow | 4.8127 | -0.574938 | 142215 | 1.6638 | 361794 | 0.002764 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| BTC | BTC-USDT-SWAP | on_chain_flow | 3.0455 | -0.628661 | 149039 | 0.0157 | 337567 | 0.002962 | okx_thin_volume_watch | OKX 24h volume proxy is low for repeat observation |
| MON | MON-USDT-SWAP | on_chain_flow | 3.8131 | 0.438000 | 412853570 | 4.4733 | 2557 | 0.391140 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| STX | STX-USDT-SWAP | on_chain_flow | 3.4022 | -0.392296 | 4529944 | 5.3064 | 1340 | 0.746520 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| BERA | BERA-USDT-SWAP | on_chain_flow | 3.2612 | -0.872475 | 24245846 | 3.8439 | 2536 | 0.394274 | okx_thin_depth_watch | 1k notional uses too much visible OKX 10 bps depth |
| OP | OP-USDT-SWAP | on_chain_flow | 2.9303 | 0.876000 | 137861046 | 10.2093 | 8292 | 0.120597 | okx_wide_spread_watch | OKX current spread is wide for a small repeat |

## Interpretation

OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only means the public venue context does not obviously block a small repeat observation; account fees, fill quality, and operational constraints are still unchecked.
