# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ZEC | broad_alpha_paper | 368.0003 | -1.878631 | 314951045 | 0.6652 | 86244 | 0.011595 | tradable_context_ok | public venue context does not obviously block a small repeat |
| INJ | broad_alpha_paper | 93.1248 | 0.013330 | 4798229 | 6.9535 | 4311 | 0.231980 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ZRO | broad_alpha_paper | 87.8705 | 0.109500 | 2473759 | 2.8994 | 9789 | 0.102151 | tradable_context_ok | public venue context does not obviously block a small repeat |
| FET | broad_alpha_paper | 77.6754 | 0.109500 | 1864412 | 1.9362 | 8287 | 0.120672 | tradable_context_ok | public venue context does not obviously block a small repeat |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 86753018 | 1.4417 | 48219 | 0.020739 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.202001 | 244902417 | 0.1527 | 446839 | 0.002238 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | -0.052196 | 6097153 | 2.0184 | 118105 | 0.008467 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | -0.088799 | 1270683830 | 1.2022 | 6111043 | 0.000164 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | -0.227003 | 2808617 | 2.4860 | 18916 | 0.052865 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 34472741 | 3.5349 | 57408 | 0.017419 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.416944 | 10528862 | 2.3981 | 55453 | 0.018033 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.198869 | 2556245 | 1.5445 | 5027 | 0.198919 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.066072 | 2831504693 | 0.1601 | 5308438 | 0.000188 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 1025398146 | 0.3210 | 122057 | 0.008193 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 7521152 | 6.1090 | 11594 | 0.086249 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.109500 | 91272154 | 6.5309 | 22664 | 0.044122 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 862473 | 8.2618 | 1855 | 0.538986 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | 0.100025 | 801472 | 8.9378 | 2638 | 0.379048 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.109500 | 176351 | 7.1466 | 2588 | 0.386410 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.033104 | 879572 | 6.0052 | 1668 | 0.599434 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | -0.124689 | 983703 | 4.2721 | 6243 | 0.160169 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.050974 | 470651 | 6.6419 | 4308 | 0.232148 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | -0.093215 | 236737 | 2.4032 | 6797 | 0.147115 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1961126 | 7.8774 | 1919 | 0.521014 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| MON | on_chain_flow | 3.8131 | 0.109500 | 2003423 | 3.4448 | 3736 | 0.267652 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| STRK | on_chain_flow | 3.3542 | -0.120810 | 1042999 | 5.9880 | 862 | 1.159652 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| ALLO | broad_alpha_paper | 50.3715 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| BEAT | broad_alpha_paper | 7.8112 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
