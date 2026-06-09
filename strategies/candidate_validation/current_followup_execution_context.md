# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ZEC | broad_alpha_paper | 368.0003 | -1.976994 | 314447898 | 0.2190 | 56581 | 0.017674 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ZRO | broad_alpha_paper | 87.8705 | 0.109500 | 2442457 | 2.7315 | 7924 | 0.126202 | tradable_context_ok | public venue context does not obviously block a small repeat |
| FET | broad_alpha_paper | 77.6754 | 0.109500 | 1845286 | 1.9131 | 14986 | 0.066728 | tradable_context_ok | public venue context does not obviously block a small repeat |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 82106829 | 0.9409 | 12422 | 0.080502 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.158401 | 240244246 | 0.1501 | 452513 | 0.002210 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | -0.100298 | 6096244 | 0.3329 | 104825 | 0.009540 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | -0.258850 | 1174643517 | 1.1867 | 7144212 | 0.000140 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | 0.109500 | 1950269 | 3.3718 | 7232 | 0.138273 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | -0.065725 | 2761113 | 2.4423 | 24497 | 0.040821 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 31209278 | 0.1326 | 62441 | 0.016015 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.366623 | 10439270 | 1.1752 | 87497 | 0.011429 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.097175 | 2471492 | 2.9846 | 13126 | 0.076182 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.047999 | 2796828474 | 0.1587 | 3448421 | 0.000290 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 1038760136 | 0.1564 | 79599 | 0.012563 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.096703 | 85108947 | 0.4012 | 14976 | 0.066776 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 873284 | 7.1897 | 6045 | 0.165422 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | 0.109500 | 792309 | 4.6868 | 12734 | 0.078531 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.109500 | 176154 | 4.3346 | 3777 | 0.264740 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | 0.036294 | 990780 | 2.9399 | 1483 | 0.674164 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.107514 | 821403 | 1.9532 | 2105 | 0.474986 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | -0.380028 | 903699 | 2.0986 | 1397 | 0.715725 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.028372 | 457701 | 2.6957 | 5411 | 0.184794 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | -0.017610 | 239726 | 2.3765 | 9695 | 0.103147 | thin_volume_watch | 24h notional volume is low for repeat observation |
| INJ | broad_alpha_paper | 93.1248 | 0.086769 | 4726032 | 2.3127 | 3198 | 0.312708 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1931060 | 7.2288 | 3340 | 0.299410 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 7391838 | 1.2695 | 3803 | 0.262941 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| ALLO | broad_alpha_paper | 50.3715 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| BEAT | broad_alpha_paper | 7.8112 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
