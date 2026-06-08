# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 82497719 | 2.2935 | 15509 | 0.064478 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.066307 | 273521903 | 0.1487 | 490438 | 0.002039 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.085721 | 7435720 | 0.1651 | 85948 | 0.011635 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | -0.225397 | 1208580568 | 0.5932 | 12920198 | 0.000077 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | 0.109500 | 2562158 | 1.8689 | 4700 | 0.212745 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | -0.087696 | 2771223 | 3.6217 | 14904 | 0.067096 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 38148936 | 0.7901 | 61586 | 0.016238 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.227713 | 11970291 | 1.7602 | 115924 | 0.008626 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.166143 | 2563174 | 2.9472 | 18330 | 0.054554 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.098684 | 3215064309 | 0.1578 | 4823106 | 0.000207 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 1029990096 | 0.1569 | 84402 | 0.011848 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 7422298 | 0.3173 | 12113 | 0.082554 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.109500 | 75776001 | 3.3309 | 26694 | 0.037461 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 814912 | 6.1988 | 4041 | 0.247477 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.436314 | 817610 | 2.4029 | 11461 | 0.087254 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.092027 | 179108 | 3.7490 | 2450 | 0.408145 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | 0.109500 | 869281 | 5.7554 | 13970 | 0.071584 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.109500 | 820105 | 4.1295 | 6792 | 0.147241 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | 0.037869 | 804663 | 9.2683 | 1234 | 0.810564 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | -0.002058 | 630694 | 4.9327 | 4981 | 0.200770 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | 0.109500 | 244461 | 0.9069 | 11511 | 0.086871 | thin_volume_watch | 24h notional volume is low for repeat observation |
| ZRO | broad_alpha_paper | 216.0244 | 0.205175 | 2963137 | 5.5790 | 582 | 1.716825 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1987331 | 6.1114 | 2409 | 0.415039 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
