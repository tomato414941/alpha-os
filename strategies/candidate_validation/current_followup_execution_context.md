# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ZEC | broad_alpha_paper | 368.0003 | -1.763160 | 316072212 | 0.2165 | 58689 | 0.017039 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ZRO | broad_alpha_paper | 87.8705 | 0.109500 | 3078954 | 2.2361 | 8951 | 0.111720 | tradable_context_ok | public venue context does not obviously block a small repeat |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 83050366 | 3.2905 | 29459 | 0.033946 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.052613 | 248395213 | 0.1494 | 408705 | 0.002447 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.090210 | 6069680 | 0.1655 | 90292 | 0.011075 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | -0.099810 | 1128474095 | 0.5899 | 11600383 | 0.000086 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | 0.109500 | 2275006 | 2.8354 | 4514 | 0.221526 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | 0.032387 | 2712383 | 2.4304 | 9948 | 0.100527 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 33336573 | 1.9811 | 49867 | 0.020053 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.365797 | 10558827 | 2.9145 | 108713 | 0.009199 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | 0.089102 | 2419289 | 4.4500 | 10844 | 0.092218 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.058396 | 2848792491 | 0.1581 | 1963090 | 0.000509 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 1053097186 | 0.1582 | 136081 | 0.007349 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 7415340 | 5.0089 | 18167 | 0.055044 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | -0.074305 | 86739180 | 4.8206 | 4171 | 0.239761 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 864037 | 4.1421 | 6654 | 0.150275 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.067406 | 799997 | 4.8479 | 15802 | 0.063285 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.109500 | 170154 | 7.0219 | 3787 | 0.264072 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | -0.141312 | 950428 | 8.7374 | 5532 | 0.180767 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.109500 | 830570 | 8.1004 | 1339 | 0.746985 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | 0.052391 | 784345 | 9.3259 | 1529 | 0.654172 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.059959 | 478250 | 5.0958 | 7486 | 0.133588 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | -0.062369 | 235314 | 2.9159 | 9935 | 0.100654 | thin_volume_watch | 24h notional volume is low for repeat observation |
| INJ | broad_alpha_paper | 93.1248 | -0.260868 | 4754218 | 4.5362 | 1599 | 0.625559 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| FET | broad_alpha_paper | 77.6754 | 0.109500 | 1871035 | 5.6958 | 3332 | 0.300129 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1907375 | 5.1353 | 1365 | 0.732473 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| ALLO | broad_alpha_paper | 50.3715 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| BEAT | broad_alpha_paper | 7.8112 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
