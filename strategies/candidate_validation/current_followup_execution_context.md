# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 69450412 | 5.2415 | 11665 | 0.085727 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.162707 | 336821754 | 0.1526 | 723180 | 0.001383 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.109500 | 10331173 | 1.3428 | 88346 | 0.011319 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | -0.008981 | 989239713 | 0.6015 | 10621101 | 0.000094 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | 0.096185 | 2204230 | 3.6753 | 21859 | 0.045748 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 44310263 | 1.6145 | 62554 | 0.015986 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.039375 | 12022601 | 0.6205 | 50204 | 0.019919 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.159211 | 1798858 | 1.5114 | 16741 | 0.059735 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.092945 | 2943217878 | 0.1589 | 3663105 | 0.000273 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 750995437 | 0.3202 | 113501 | 0.008810 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.485862 | 8501930 | 0.9924 | 8004 | 0.124930 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.109500 | 76274821 | 3.5875 | 10352 | 0.096603 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 547356 | 5.8749 | 1959 | 0.510434 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.396361 | 719584 | 7.5496 | 15009 | 0.066628 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | -0.015571 | 196225 | 5.9116 | 4143 | 0.241364 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | 0.109500 | 820578 | 5.9506 | 5630 | 0.177624 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.091772 | 262331 | 4.1925 | 3649 | 0.274014 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | -0.076821 | 871144 | 5.2228 | 2930 | 0.341300 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.041428 | 583883 | 4.0486 | 6925 | 0.144403 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | 0.109500 | 206398 | 6.7952 | 7750 | 0.129035 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.042638 | 1828465 | 9.6351 | 1965 | 0.508978 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| MON | on_chain_flow | 3.8131 | 0.109500 | 3947965 | 3.6738 | 3786 | 0.264162 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
