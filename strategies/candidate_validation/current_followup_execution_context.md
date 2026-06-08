# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 72674861 | 6.0695 | 7339 | 0.136265 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.201950 | 338967398 | 0.9076 | 477326 | 0.002095 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.109500 | 9795154 | 1.1743 | 61067 | 0.016375 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | 0.063046 | 991384276 | 0.5980 | 9957983 | 0.000100 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | 0.109500 | 3747019 | 2.2870 | 5334 | 0.187494 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | 0.020488 | 1907572 | 3.6476 | 49204 | 0.020324 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 50082572 | 2.4981 | 45836 | 0.021817 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | 0.061895 | 11423568 | 0.6108 | 54861 | 0.018228 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.233827 | 1674882 | 2.9994 | 11879 | 0.084184 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.051997 | 2912505450 | 0.3160 | 4144791 | 0.000241 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 773319745 | 0.1587 | 82235 | 0.012160 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 8631829 | 3.8168 | 4455 | 0.224464 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.109500 | 76215790 | 4.0267 | 9980 | 0.100203 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 518513 | 6.4547 | 4833 | 0.206895 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.042251 | 789951 | 3.0182 | 6862 | 0.145726 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.109500 | 190333 | 5.8715 | 4026 | 0.248385 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | -0.003062 | 617439 | 2.9546 | 1789 | 0.559055 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.109500 | 261657 | 4.5565 | 4987 | 0.200514 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | -0.083556 | 850199 | 5.1592 | 1374 | 0.727606 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | -0.084242 | 529326 | 2.2676 | 8075 | 0.123837 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | 0.109500 | 212747 | 4.5653 | 10752 | 0.093005 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1821058 | 8.2764 | 2899 | 0.344968 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
