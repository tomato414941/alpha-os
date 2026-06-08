# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NEAR | exchange_catalyst;on_chain_flow | 8.3498 | 0.109500 | 82501013 | 0.4599 | 52721 | 0.018968 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | exchange_catalyst;on_chain_flow | 5.2019 | -0.067406 | 275339004 | 0.1487 | 540957 | 0.001849 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BNB | l2_imbalance;on_chain_flow | 4.8127 | 0.083337 | 7474718 | 1.1554 | 90614 | 0.011036 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MEGA | exchange_catalyst;on_chain_flow | 4.4940 | 0.109500 | 1991128 | 5.5305 | 4370 | 0.228833 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | on_chain_flow | 3.9347 | -0.229776 | 1211252909 | 0.5932 | 12753109 | 0.000078 | tradable_context_ok | public venue context does not obviously block a small repeat |
| MON | on_chain_flow | 3.8131 | 0.109500 | 2572102 | 2.3348 | 4618 | 0.216534 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ARB | on_chain_flow | 3.7070 | -0.076163 | 2776279 | 1.2066 | 5332 | 0.187538 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SUI | on_chain_flow | 3.6307 | 0.109500 | 38205497 | 1.4473 | 75195 | 0.013299 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ADA | on_chain_flow | 3.4842 | -0.215047 | 12014774 | 1.1720 | 86910 | 0.011506 | tradable_context_ok | public venue context does not obviously block a small repeat |
| APT | on_chain_flow | 3.3984 | -0.172336 | 2577942 | 1.4735 | 11285 | 0.088617 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | on_chain_flow | 3.0455 | 0.099741 | 3234377150 | 0.1578 | 1354031 | 0.000739 | tradable_context_ok | public venue context does not obviously block a small repeat |
| HYPE | on_chain_flow | 2.6935 | 0.109500 | 1031970449 | 0.1569 | 65656 | 0.015231 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XMR | l2_imbalance | 2.3874 | 0.109500 | 7491317 | 1.9036 | 7612 | 0.131380 | tradable_context_ok | public venue context does not obviously block a small repeat |
| WLD | l2_imbalance | 2.3679 | 0.109500 | 75902800 | 0.3686 | 11951 | 0.083673 | tradable_context_ok | public venue context does not obviously block a small repeat |
| CHIP | exchange_catalyst | 3.9957 | 0.109500 | 790180 | 3.2540 | 3568 | 0.280291 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SEI | on_chain_flow | 3.8994 | -0.418396 | 818513 | 3.4030 | 9942 | 0.100588 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STX | on_chain_flow | 3.4022 | 0.109500 | 180786 | 4.2843 | 3819 | 0.261879 | thin_volume_watch | 24h notional volume is low for repeat observation |
| STRK | on_chain_flow | 3.3542 | 0.109500 | 875461 | 5.7554 | 9761 | 0.102450 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BERA | on_chain_flow | 3.2612 | 0.109500 | 821286 | 4.5094 | 4465 | 0.223953 | thin_volume_watch | 24h notional volume is low for repeat observation |
| OP | on_chain_flow | 2.9303 | 0.051278 | 806260 | 9.2779 | 5036 | 0.198586 | thin_volume_watch | 24h notional volume is low for repeat observation |
| POL | sector_perp_context;on_chain_flow | 2.7572 | 0.026509 | 630682 | 2.2757 | 5652 | 0.176930 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MNT | on_chain_flow | 1.7848 | 0.109500 | 244538 | 1.8134 | 17101 | 0.058477 | thin_volume_watch | 24h notional volume is low for repeat observation |
| PEPE | liquidation | 2.1830 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
