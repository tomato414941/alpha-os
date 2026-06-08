# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 73003116 | 3.2492 | 16688 | 0.059924 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.137424 | 316334563 | 0.1517 | 401209 | 0.002492 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.109500 | 9412339 | 0.1674 | 86564 | 0.011552 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | 0.028238 | 967264490 | 0.6020 | 13326933 | 0.000075 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | -0.091975 | 3622587 | 2.2990 | 4836 | 0.206762 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 51198691 | 1.1985 | 52790 | 0.018943 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.081288 | 9029050 | 1.8330 | 93065 | 0.010745 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.403047 | 1871257 | 4.5334 | 7553 | 0.132392 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.109500 | 2950516658 | 0.1579 | 4073664 | 0.000245 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 770511645 | 0.1629 | 66313 | 0.015080 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 8390319 | 2.5367 | 10760 | 0.092936 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.109500 | 75477920 | 2.1280 | 26062 | 0.038370 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 536760 | 3.4746 | 7102 | 0.140807 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.078826 | 760715 | 6.2325 | 15011 | 0.066618 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.109500 | 188440 | 3.2160 | 3975 | 0.251600 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | 0.057028 | 640726 | 5.8945 | 5069 | 0.197283 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.093064 | 345113 | 3.3040 | 3609 | 0.277111 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | -0.042290 | 873980 | 8.2910 | 1196 | 0.836462 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.109500 | 519568 | 4.0398 | 6686 | 0.149558 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | 0.109500 | 234888 | 6.2047 | 11724 | 0.085297 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | -0.322368 | 2174227 | 9.1448 | 576 | 1.735524 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| ARB | on_chain_flow | 3.7070 | -0.223367 | 2048247 | 3.6735 | 3007 | 0.332570 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
