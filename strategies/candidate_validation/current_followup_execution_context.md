# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ZRO | broad_alpha_paper | 216.0244 | 0.200706 | 2966014 | 2.7959 | 4054 | 0.246670 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ZEC | broad_alpha_paper | 215.1219 | -0.918285 | 329074511 | 0.4273 | 156874 | 0.006375 | tradable_context_ok | public venue context does not obviously block a small repeat |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 83452994 | 5.4583 | 9820 | 0.101828 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.061810 | 274851428 | 0.1483 | 321253 | 0.003113 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.093056 | 7438973 | 0.9895 | 84715 | 0.011804 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | -0.216566 | 1213787135 | 0.5923 | 12425137 | 0.000080 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | 0.109500 | 2568975 | 3.7227 | 5584 | 0.179080 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | -0.092907 | 2783766 | 2.4079 | 8913 | 0.112198 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 38152661 | 1.3145 | 66829 | 0.014964 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.225960 | 11977033 | 2.3477 | 86362 | 0.011579 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.177213 | 2568241 | 1.4705 | 8412 | 0.118876 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.101314 | 3220483690 | 0.1576 | 3306171 | 0.000302 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 1031147883 | 0.1566 | 161771 | 0.006182 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 7424589 | 0.3170 | 16362 | 0.061116 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.109500 | 76960207 | 4.0443 | 16631 | 0.060129 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 816935 | 4.9942 | 2596 | 0.385210 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.459958 | 818789 | 3.0042 | 11107 | 0.090036 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.074427 | 179416 | 4.8096 | 5215 | 0.191745 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | 0.109500 | 871494 | 5.7389 | 4636 | 0.215696 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.109500 | 820366 | 4.8875 | 4622 | 0.216375 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | 0.014385 | 805066 | 6.1665 | 5739 | 0.174233 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | -0.030956 | 633659 | 2.5330 | 5181 | 0.193019 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | 0.109500 | 244652 | 1.9964 | 15349 | 0.065153 | thin_volume_watch | 24h notional volume is low for repeat observation |
| EIGEN | broad_alpha_paper | 48.9663 | 0.109500 | 1475017 | 5.4422 | 1364 | 0.733378 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1988446 | 3.5482 | 3994 | 0.250406 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| BEAT | broad_alpha_paper | 25.2717 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
