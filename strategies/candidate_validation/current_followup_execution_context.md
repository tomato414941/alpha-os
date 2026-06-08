# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ZEC | broad_alpha_paper | 222.5474 | -1.335783 | 334245273 | 1.8967 | 120252 | 0.008316 | tradable_context_ok | public venue context does not obviously block a small repeat |
| INJ | broad_alpha_paper | 56.7834 | -0.295351 | 4987471 | 2.5972 | 6019 | 0.166133 | tradable_context_ok | public venue context does not obviously block a small repeat |
| FET | broad_alpha_paper | 37.4339 | 0.109500 | 1792502 | 7.8881 | 6110 | 0.163659 | tradable_context_ok | public venue context does not obviously block a small repeat |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 84748448 | 5.5198 | 25236 | 0.039626 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | 0.109500 | 278904354 | 0.1472 | 365054 | 0.002739 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.109500 | 7425315 | 0.1639 | 82408 | 0.012135 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1988352 | 3.3410 | 4623 | 0.216328 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | 0.060914 | 1242764538 | 0.5846 | 9944892 | 0.000101 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | 0.109500 | 2530258 | 3.7121 | 4207 | 0.237687 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | -0.099812 | 2795343 | 3.5814 | 11759 | 0.085041 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 38191136 | 0.1304 | 75509 | 0.013243 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.287848 | 12075918 | 3.4754 | 73077 | 0.013684 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.126105 | 2603583 | 4.3812 | 10495 | 0.095287 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.109500 | 3233354751 | 0.1569 | 3010901 | 0.000332 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 1035920900 | 0.1563 | 238708 | 0.004189 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 7430868 | 2.5229 | 11100 | 0.090087 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.109500 | 77176706 | 0.1849 | 32813 | 0.030475 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.085702 | 863145 | 9.0462 | 4193 | 0.238509 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.349499 | 821418 | 3.7859 | 7363 | 0.135822 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.026570 | 188392 | 6.3721 | 2861 | 0.349480 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | 0.109500 | 979616 | 8.5874 | 1328 | 0.753011 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | -0.004373 | 821067 | 7.1240 | 2174 | 0.459921 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | -0.099530 | 816194 | 5.0883 | 1641 | 0.609488 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.019277 | 636376 | 3.2759 | 4569 | 0.218871 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | 0.109500 | 244972 | 2.8923 | 11655 | 0.085800 | thin_volume_watch | 24h notional volume is low for repeat observation |
| ZRO | broad_alpha_paper | 204.4985 | 0.109500 | 3004528 | 4.5995 | 1061 | 0.942443 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| EIGEN | broad_alpha_paper | 5.4407 | 0.109500 | 1480352 | 10.8284 | 326 | 3.064783 | wide_spread_watch | current spread is wide for a small directional repeat |
| DEUS | broad_alpha_paper | 174.3031 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
