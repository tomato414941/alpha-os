# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 72773809 | 2.3167 | 12910 | 0.077457 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.117458 | 303069550 | 0.1510 | 691489 | 0.001446 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.109500 | 9377236 | 2.1707 | 90887 | 0.011003 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | 0.034077 | 961921868 | 0.6002 | 11765011 | 0.000085 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | -0.170628 | 2059758 | 3.6570 | 10294 | 0.097141 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 50979503 | 1.4617 | 52474 | 0.019057 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.062735 | 9033849 | 3.6543 | 76257 | 0.013114 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.386306 | 1976439 | 2.9985 | 10794 | 0.092644 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.109500 | 2972330224 | 0.1577 | 1819326 | 0.000550 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 766770492 | 0.6508 | 92883 | 0.010766 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 8322863 | 2.8559 | 10573 | 0.094580 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.109500 | 76170186 | 7.8771 | 29447 | 0.033959 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.096620 | 536949 | 5.0704 | 6502 | 0.153808 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.107971 | 779616 | 3.5960 | 12042 | 0.083040 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.109500 | 191259 | 4.2769 | 4207 | 0.237715 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | -0.009352 | 636347 | 2.9304 | 3492 | 0.286382 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.097382 | 353462 | 9.3195 | 1097 | 0.911246 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | -0.061476 | 829287 | 6.1773 | 1691 | 0.591236 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.109500 | 520637 | 5.3972 | 1680 | 0.595319 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | 0.109500 | 237628 | 5.1079 | 8634 | 0.115821 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | -0.082874 | 2148845 | 11.0654 | 1123 | 0.890599 | wide_spread_watch | current spread is wide for a small directional repeat |
| MON | on_chain_flow | 3.8131 | -0.060011 | 3564021 | 3.2039 | 2257 | 0.443102 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
