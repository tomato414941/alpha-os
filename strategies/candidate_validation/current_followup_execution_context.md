# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 71868339 | 2.7527 | 57413 | 0.017418 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.316550 | 337521134 | 0.1521 | 224820 | 0.004448 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.109500 | 9897685 | 1.1751 | 86335 | 0.011583 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | -0.000950 | 966996396 | 0.5989 | 12958184 | 0.000077 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | 0.006147 | 1915897 | 3.6623 | 10300 | 0.097085 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 43874657 | 0.1337 | 75354 | 0.013271 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | 0.036465 | 11221064 | 0.6170 | 65229 | 0.015331 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.204665 | 1677826 | 1.5032 | 13768 | 0.072632 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.020223 | 2891590935 | 0.1584 | 2407309 | 0.000415 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 764372490 | 0.1586 | 110716 | 0.009032 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.109500 | 76056420 | 1.8773 | 22759 | 0.043938 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 524661 | 6.1587 | 6327 | 0.158050 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.223789 | 809287 | 5.2822 | 12952 | 0.077210 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.022626 | 189174 | 3.2156 | 4304 | 0.232337 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | -0.003379 | 622981 | 2.9731 | 1634 | 0.612032 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.109500 | 262980 | 6.6733 | 2725 | 0.366944 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | -0.345703 | 866853 | 7.2490 | 6794 | 0.147187 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | -0.164854 | 541252 | 7.0838 | 4199 | 0.238159 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | 0.109500 | 210318 | 6.3898 | 8098 | 0.123487 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1824682 | 5.6822 | 2483 | 0.402704 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| MON | on_chain_flow | 3.8131 | 0.072062 | 3782708 | 6.4117 | 3257 | 0.307046 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 8401514 | 1.6085 | 2113 | 0.473303 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
