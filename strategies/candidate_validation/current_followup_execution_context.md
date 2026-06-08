# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ZEC | broad_alpha_paper | 368.0003 | -1.399200 | 335885572 | 2.1129 | 100255 | 0.009975 | tradable_context_ok | public venue context does not obviously block a small repeat |
| INJ | broad_alpha_paper | 93.1248 | -0.284489 | 5029234 | 1.5533 | 4239 | 0.235901 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ZRO | broad_alpha_paper | 87.8705 | 0.109500 | 3009819 | 4.7240 | 5262 | 0.190047 | tradable_context_ok | public venue context does not obviously block a small repeat |
| FET | broad_alpha_paper | 77.6754 | 0.109500 | 1793693 | 3.2533 | 9335 | 0.107120 | tradable_context_ok | public venue context does not obviously block a small repeat |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 84801202 | 2.3064 | 20508 | 0.048761 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | 0.109500 | 279292013 | 0.1472 | 383025 | 0.002611 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.109500 | 7427422 | 0.1641 | 68678 | 0.014561 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1988352 | 2.9439 | 5485 | 0.182307 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | 0.067479 | 1244248849 | 0.5843 | 10191872 | 0.000098 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | -0.088513 | 2797513 | 2.3898 | 5993 | 0.166857 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 38258130 | 0.1302 | 103817 | 0.009632 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.287661 | 12094863 | 2.3175 | 104756 | 0.009546 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.107863 | 2607309 | 2.9218 | 11110 | 0.090009 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.109500 | 3235063931 | 0.1570 | 399102 | 0.002506 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 1038465303 | 0.1561 | 46380 | 0.021561 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 7442995 | 1.5800 | 4652 | 0.214952 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.109500 | 78873769 | 7.2212 | 20715 | 0.048274 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 865839 | 1.4587 | 999 | 1.000743 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.266719 | 821659 | 3.9855 | 12104 | 0.082620 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.015729 | 189442 | 4.2474 | 3433 | 0.291278 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | 0.109500 | 979829 | 8.5898 | 3463 | 0.288760 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.051065 | 821937 | 2.6314 | 3753 | 0.266475 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | -0.044411 | 816748 | 7.1258 | 5477 | 0.182580 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.001209 | 636456 | 2.3984 | 8180 | 0.122255 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | 0.109500 | 245130 | 2.7143 | 11174 | 0.089495 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MON | on_chain_flow | 3.8131 | 0.109500 | 2530666 | 5.5623 | 3136 | 0.318913 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| DEUS | broad_alpha_paper | 174.3031 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| ALLO | broad_alpha_paper | 50.3715 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| BEAT | broad_alpha_paper | 7.8112 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
