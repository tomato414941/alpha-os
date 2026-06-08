# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 72817295 | 1.8321 | 30119 | 0.033202 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.186182 | 299588778 | 0.1510 | 391484 | 0.002554 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.109500 | 9306752 | 1.1688 | 87369 | 0.011446 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | 0.050644 | 960067515 | 0.5993 | 12978614 | 0.000077 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | 0.109500 | 3578478 | 0.4592 | 5594 | 0.178774 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | -0.061552 | 2036791 | 1.2116 | 6273 | 0.159426 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 50246088 | 1.3293 | 80606 | 0.012406 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.024010 | 9061186 | 2.4207 | 68911 | 0.014511 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.126162 | 2006626 | 2.9815 | 11540 | 0.086658 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.109500 | 2979459099 | 0.1577 | 537001 | 0.001862 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 760191811 | 0.8139 | 45067 | 0.022189 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 8361103 | 3.4922 | 7671 | 0.130359 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | -0.016141 | 75689653 | 2.3348 | 31532 | 0.031713 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.047220 | 540452 | 4.1311 | 4889 | 0.204547 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.175893 | 780993 | 2.0010 | 7722 | 0.129508 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.109500 | 191836 | 2.6714 | 3823 | 0.261602 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | 0.023544 | 625891 | 8.6969 | 2968 | 0.336977 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.109500 | 396402 | 4.4117 | 1939 | 0.515729 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | -0.401900 | 831490 | 3.0733 | 982 | 1.018454 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.109500 | 519450 | 6.0184 | 6458 | 0.154850 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | 0.109500 | 230273 | 2.5503 | 11343 | 0.088161 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 2120730 | 14.1515 | 890 | 1.123542 | wide_spread_watch | current spread is wide for a small directional repeat |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
