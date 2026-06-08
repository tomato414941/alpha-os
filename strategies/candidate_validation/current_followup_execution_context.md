# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 72985211 | 4.7004 | 38661 | 0.025866 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.138587 | 339480790 | 0.1518 | 819292 | 0.001221 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.109500 | 9826387 | 0.1682 | 126161 | 0.007926 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | 0.087719 | 997872768 | 0.5999 | 12804254 | 0.000078 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | 0.045750 | 3750234 | 2.2956 | 4407 | 0.226895 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | -0.025227 | 1911246 | 1.2197 | 6567 | 0.152268 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 50126303 | 1.0562 | 40216 | 0.024866 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.154862 | 11467478 | 2.4529 | 75550 | 0.013236 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | 0.109500 | 1684188 | 1.5036 | 5825 | 0.171662 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.006948 | 2918094280 | 0.1584 | 3249885 | 0.000308 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 774379241 | 0.1586 | 42251 | 0.023668 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.038749 | 8667562 | 2.2390 | 10047 | 0.099537 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.016812 | 76312947 | 1.6923 | 12541 | 0.079736 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | -0.236269 | 519805 | 3.8856 | 4862 | 0.205666 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | 0.109500 | 794150 | 0.6040 | 5355 | 0.186747 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.109500 | 192709 | 2.1403 | 3892 | 0.256908 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | 0.109500 | 629075 | 5.9330 | 9385 | 0.106557 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.109500 | 263718 | 3.3285 | 919 | 1.087909 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | 0.109500 | 852949 | 7.2490 | 2009 | 0.497723 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.031906 | 535022 | 4.8011 | 5535 | 0.180667 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | 0.109500 | 231362 | 4.9408 | 4873 | 0.205231 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1881469 | 10.8525 | 494 | 2.023571 | wide_spread_watch | current spread is wide for a small directional repeat |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
