# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ZEC | broad_alpha_paper | 368.0003 | -1.408027 | 329806599 | 0.2124 | 36785 | 0.027185 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ZRO | broad_alpha_paper | 87.8705 | 0.109500 | 3037631 | 6.6454 | 6326 | 0.158076 | tradable_context_ok | public venue context does not obviously block a small repeat |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 84745917 | 2.7889 | 22274 | 0.044895 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | 0.066561 | 267711992 | 0.2975 | 403599 | 0.002478 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.032462 | 6367805 | 1.1542 | 95935 | 0.010424 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | -0.112927 | 1197356662 | 0.5900 | 12405452 | 0.000081 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | 0.109500 | 2383982 | 3.2766 | 6018 | 0.166178 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | 0.109500 | 2760409 | 3.6169 | 7220 | 0.138505 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 37764004 | 1.8389 | 72465 | 0.013800 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.310369 | 11731415 | 3.5082 | 95328 | 0.010490 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | 0.109500 | 2512656 | 2.9477 | 15030 | 0.066536 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.025767 | 3094844846 | 0.1576 | 2495461 | 0.000401 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 1044809075 | 3.6142 | 88943 | 0.011243 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 7305353 | 0.3142 | 10165 | 0.098376 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.109500 | 84563299 | 6.0482 | 12949 | 0.077226 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 911223 | 4.7192 | 3992 | 0.250487 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.133138 | 810674 | 2.6108 | 8196 | 0.122010 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.109500 | 176166 | 6.9685 | 3427 | 0.291793 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | -0.206118 | 966124 | 5.8106 | 8170 | 0.122398 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.109500 | 826768 | 3.4373 | 2766 | 0.361533 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | 0.109500 | 797577 | 9.2502 | 202 | 4.952547 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | -0.096505 | 602235 | 2.1593 | 7586 | 0.131829 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | 0.009564 | 242996 | 3.0877 | 13495 | 0.074103 | thin_volume_watch | 24h notional volume is low for repeat observation |
| INJ | broad_alpha_paper | 93.1248 | -0.235013 | 4939224 | 7.4659 | 3953 | 0.252987 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| FET | broad_alpha_paper | 77.6754 | 0.109500 | 1816089 | 6.1244 | 3446 | 0.290163 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1965791 | 6.9354 | 2149 | 0.465393 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| ALLO | broad_alpha_paper | 50.3715 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| BEAT | broad_alpha_paper | 7.8112 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
