# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ZEC | broad_alpha_paper | 368.0003 | -1.776333 | 315535232 | 0.2173 | 139491 | 0.007169 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ZRO | broad_alpha_paper | 87.8705 | 0.109500 | 3076638 | 2.2365 | 5769 | 0.173328 | tradable_context_ok | public venue context does not obviously block a small repeat |
| FET | broad_alpha_paper | 77.6754 | 0.109500 | 1865689 | 3.7843 | 11616 | 0.086085 | tradable_context_ok | public venue context does not obviously block a small repeat |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 83009288 | 3.7520 | 31626 | 0.031620 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.041207 | 248199545 | 0.1492 | 483978 | 0.002066 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.082940 | 6066872 | 0.1653 | 84983 | 0.011767 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1907027 | 3.3520 | 5006 | 0.199775 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | -0.107611 | 1126154126 | 0.5900 | 12447193 | 0.000080 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | 0.109500 | 2258553 | 5.6542 | 6248 | 0.160044 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | 0.032338 | 2711204 | 2.4254 | 24046 | 0.041586 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 33321282 | 0.1319 | 58025 | 0.017234 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.362636 | 10537259 | 2.9160 | 79638 | 0.012557 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | 0.109500 | 2419206 | 4.4421 | 9329 | 0.107194 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.051649 | 2845847674 | 0.1581 | 3197910 | 0.000313 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 1052384622 | 0.1582 | 98944 | 0.010107 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | -0.054112 | 86520028 | 1.8055 | 14967 | 0.066812 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 863611 | 4.4144 | 6149 | 0.162623 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.070351 | 794011 | 6.2575 | 14999 | 0.066671 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.109500 | 170021 | 6.4756 | 2726 | 0.366900 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | -0.152175 | 949867 | 5.8106 | 3601 | 0.277714 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.109500 | 830461 | 6.1629 | 2069 | 0.483316 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | 0.072182 | 784145 | 7.2453 | 5604 | 0.178453 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.075290 | 478233 | 3.9485 | 7172 | 0.139423 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | -0.078226 | 235188 | 1.0944 | 5764 | 0.173498 | thin_volume_watch | 24h notional volume is low for repeat observation |
| INJ | broad_alpha_paper | 93.1248 | -0.282887 | 4753943 | 5.5743 | 3072 | 0.325543 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 7298716 | 1.5678 | 3349 | 0.298632 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| ALLO | broad_alpha_paper | 50.3715 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| BEAT | broad_alpha_paper | 7.8112 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
