# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ZEC | broad_alpha_paper | 368.0003 | -1.994637 | 311832167 | 0.4482 | 88242 | 0.011332 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ZRO | broad_alpha_paper | 87.8705 | 0.109500 | 2437048 | 5.4469 | 4149 | 0.241027 | tradable_context_ok | public venue context does not obviously block a small repeat |
| FET | broad_alpha_paper | 77.6754 | 0.109500 | 1841473 | 1.4486 | 13662 | 0.073195 | tradable_context_ok | public venue context does not obviously block a small repeat |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 85690913 | 1.4502 | 20168 | 0.049583 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.184362 | 241087792 | 0.1522 | 461288 | 0.002168 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | -0.068400 | 6013947 | 1.1778 | 77558 | 0.012894 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | -0.128847 | 1232041564 | 0.5997 | 11418241 | 0.000088 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | -0.171488 | 2749482 | 1.2389 | 4849 | 0.206217 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 31597493 | 1.4846 | 108818 | 0.009190 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.366001 | 10444760 | 2.9934 | 55821 | 0.017914 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.180990 | 2437016 | 3.0381 | 7889 | 0.126752 | tradable_context_ok | public venue context does not obviously block a small repeat |
| STRK | on_chain_flow | 3.3542 | -0.050498 | 1034598 | 5.9827 | 4770 | 0.209647 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.060108 | 2794918382 | 0.1597 | 3580817 | 0.000279 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 997260666 | 0.1589 | 70927 | 0.014099 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.109500 | 87802321 | 0.1996 | 19015 | 0.052591 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 860482 | 6.9884 | 3750 | 0.266655 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | 0.109500 | 796541 | 3.3177 | 6546 | 0.152754 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.109500 | 174820 | 6.5923 | 2457 | 0.407081 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.095573 | 872626 | 4.7735 | 4704 | 0.212582 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | -0.176677 | 976953 | 7.4559 | 621 | 1.610909 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.059664 | 458495 | 9.3587 | 2893 | 0.345677 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | -0.053328 | 236284 | 2.0282 | 7663 | 0.130489 | thin_volume_watch | 24h notional volume is low for repeat observation |
| INJ | broad_alpha_paper | 93.1248 | 0.079147 | 4733134 | 7.9621 | 702 | 1.423731 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1950171 | 7.5953 | 3377 | 0.296101 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| MON | on_chain_flow | 3.8131 | 0.109500 | 1984107 | 2.9478 | 3451 | 0.289802 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 7458125 | 0.3220 | 481 | 2.077714 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| ALLO | broad_alpha_paper | 50.3715 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| BEAT | broad_alpha_paper | 7.8112 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
