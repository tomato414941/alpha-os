# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ZEC | broad_alpha_paper | 368.0003 | -1.809971 | 317968557 | 1.7471 | 47984 | 0.020840 | tradable_context_ok | public venue context does not obviously block a small repeat |
| INJ | broad_alpha_paper | 93.1248 | -0.278851 | 4744403 | 4.3603 | 4938 | 0.202499 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ZRO | broad_alpha_paper | 87.8705 | 0.109500 | 3066155 | 2.9439 | 6762 | 0.147877 | tradable_context_ok | public venue context does not obviously block a small repeat |
| FET | broad_alpha_paper | 77.6754 | 0.109500 | 1839281 | 1.8934 | 13315 | 0.075103 | tradable_context_ok | public venue context does not obviously block a small repeat |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 82211041 | 2.8156 | 17268 | 0.057910 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.062612 | 246553041 | 0.1492 | 395500 | 0.002528 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.102436 | 5993815 | 0.1654 | 105913 | 0.009442 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | -0.077471 | 1122726510 | 0.5884 | 11769551 | 0.000085 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | 0.109500 | 2170474 | 1.4136 | 6115 | 0.163545 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | 0.014154 | 2655845 | 2.4295 | 4013 | 0.249219 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 32757417 | 0.6596 | 70766 | 0.014131 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.383312 | 10330558 | 2.9126 | 101885 | 0.009815 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | 0.079254 | 2416616 | 2.9660 | 12239 | 0.081708 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.063334 | 2811905752 | 0.1580 | 4515571 | 0.000221 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 1043551003 | 0.1580 | 159661 | 0.006263 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 7425299 | 2.5128 | 18115 | 0.055204 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | -0.109998 | 86324665 | 0.2006 | 142049 | 0.007040 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 866269 | 4.7478 | 6991 | 0.143045 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.059086 | 797853 | 3.0254 | 15455 | 0.064704 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.109500 | 169033 | 4.8574 | 5080 | 0.196868 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | -0.145399 | 953495 | 5.8241 | 10794 | 0.092642 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.109500 | 828514 | 8.0989 | 3214 | 0.311175 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | 0.018469 | 782333 | 10.3713 | 5658 | 0.176750 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.025246 | 477294 | 4.2027 | 5630 | 0.177633 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | -0.069978 | 234914 | 2.7347 | 9875 | 0.101265 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1909661 | 9.6963 | 2569 | 0.389200 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| ALLO | broad_alpha_paper | 50.3715 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| BEAT | broad_alpha_paper | 7.8112 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
