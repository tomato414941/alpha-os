# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 81309456 | 5.0218 | 35774 | 0.027953 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | 0.027354 | 278460218 | 0.1482 | 515389 | 0.001940 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.109500 | 7478997 | 0.4935 | 93040 | 0.010748 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 2004937 | 4.5399 | 4828 | 0.207119 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | -0.134650 | 1210485300 | 0.5926 | 13070379 | 0.000077 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | 0.109500 | 2706314 | 4.6117 | 4898 | 0.204145 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | -0.044686 | 2812082 | 3.5986 | 25341 | 0.039462 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 38158706 | 0.7887 | 83185 | 0.012021 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.326921 | 12223631 | 4.6658 | 82875 | 0.012066 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.024480 | 2630056 | 5.8651 | 11788 | 0.084831 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.106565 | 3349644408 | 0.1575 | 4695936 | 0.000213 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 1041845019 | 0.1574 | 78837 | 0.012684 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.109500 | 75455329 | 7.7730 | 10931 | 0.091480 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 695365 | 4.7592 | 6045 | 0.165427 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.089256 | 838270 | 6.4209 | 7871 | 0.127054 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | -0.026537 | 183190 | 7.4563 | 2627 | 0.380602 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | 0.109500 | 867014 | 5.7372 | 8203 | 0.121906 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.109500 | 804045 | 4.4709 | 4639 | 0.215584 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | 0.109500 | 818907 | 5.1290 | 744 | 1.343390 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.109500 | 630796 | 6.1102 | 3967 | 0.252076 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | 0.109500 | 241821 | 1.9971 | 5971 | 0.167487 | thin_volume_watch | 24h notional volume is low for repeat observation |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 7234962 | 1.8921 | 1771 | 0.564753 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
