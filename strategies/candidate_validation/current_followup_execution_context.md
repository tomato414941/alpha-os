# Current Follow-Up Execution Context

This joins the follow-up queue to current Hyperliquid market context. It is a rough tradability screen, not a fill model.

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| WLD | hl_candidate;okx_pressure;liquidation | 10.0571 | -0.214318 | 65514837 | 5.3436 | 22766 | 0.043925 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | okx_pressure;liquidation;l2_imbalance | 4.5510 | 0.065543 | 643987299 | 0.6123 | 12184113 | 0.000082 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | liquidation;l2_imbalance | 3.6217 | 0.109500 | 2282491265 | 0.1607 | 2835549 | 0.000353 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ONDO | liquidation;sector_rotation | 3.6106 | 0.109500 | 15480566 | 0.8600 | 35880 | 0.027870 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XRP | okx_pressure;liquidation | 3.4627 | 0.026255 | 37464084 | 0.8741 | 525602 | 0.001903 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XPL | l2_imbalance;sector_rotation | 3.4493 | 0.109500 | 7185940 | 3.3322 | 5094 | 0.196293 | tradable_context_ok | public venue context does not obviously block a small repeat |
| LTC | okx_pressure;liquidation | 3.2959 | 0.109500 | 3164727 | 0.9519 | 55182 | 0.018122 | tradable_context_ok | public venue context does not obviously block a small repeat |
| SOL | okx_pressure;liquidation | 3.1187 | -0.193780 | 318427310 | 0.1529 | 411839 | 0.002428 | tradable_context_ok | public venue context does not obviously block a small repeat |
| PUMP | liquidation;sector_rotation | 2.9792 | 0.109500 | 3282571 | 6.6203 | 37064 | 0.026980 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XLM | okx_pressure;l2_imbalance | 2.9178 | 0.109500 | 7412753 | 6.2268 | 8613 | 0.116103 | tradable_context_ok | public venue context does not obviously block a small repeat |
| TON | okx_pressure;liquidation | 2.1872 | 0.109500 | 29108967 | 2.3301 | 16181 | 0.061799 | tradable_context_ok | public venue context does not obviously block a small repeat |
| IP | hl_candidate | 3.3166 | 0.023705 | 402581 | 3.8402 | 12576 | 0.079516 | thin_volume_watch | 24h notional volume is low for repeat observation |
| ZORA | hl_candidate | 3.2743 | -0.303753 | 142701 | 9.2894 | 449 | 2.225573 | thin_volume_watch | 24h notional volume is low for repeat observation |
| KAITO | hl_candidate | 3.1882 | 0.109500 | 119419 | 9.7531 | 1619 | 0.617611 | thin_volume_watch | 24h notional volume is low for repeat observation |
| AIXBT | hl_candidate | 3.1603 | -0.973931 | 305568 | 4.0618 | 3350 | 0.298471 | thin_volume_watch | 24h notional volume is low for repeat observation |
| APEX | hl_candidate | 3.1524 | 0.109500 | 104261 | 13.3488 | 295 | 3.393199 | thin_volume_watch | 24h notional volume is low for repeat observation |
| BSV | hl_candidate | 3.0874 | -1.435866 | 202152 | 3.3317 | 580 | 1.723395 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SAGA | hl_candidate | 3.0365 | -0.513272 | 175955 | 14.6735 | 2863 | 0.349224 | thin_volume_watch | 24h notional volume is low for repeat observation |
| PYTH | sector_rotation | 1.9270 | 0.109500 | 512542 | 3.4723 | 5184 | 0.192918 | thin_volume_watch | 24h notional volume is low for repeat observation |
| MEGA | hl_candidate | 3.8916 | -0.722928 | 1419937 | 6.4989 | 625 | 1.598890 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| XMR | hl_candidate | 3.6004 | 0.109500 | 8242547 | 1.5906 | 1234 | 0.810089 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| JTO | liquidation;l2_imbalance | 3.4579 | -0.189444 | 7412768 | 11.3558 | 4847 | 0.206330 | wide_spread_watch | current spread is wide for a small directional repeat |
| ZRO | hl_candidate | 2.5173 | 0.303161 | 2090789 | 6.8424 | 2188 | 0.457112 | thin_near_depth_watch | 1k notional uses too much visible 10 bps depth |
| PEPE | okx_pressure;liquidation | 3.7269 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| ALLO | liquidation | 3.0965 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| HOME | okx_pressure | 2.8356 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| H | liquidation | 2.7846 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| BILL | okx_pressure | 2.0283 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| EDEN | okx_pressure | 1.9688 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |
| LAB | liquidation | 1.8363 | 0.000000 | 0 |  |  |  | not_hyperliquid | asset is not in current Hyperliquid perp universe |

## Interpretation

`tradable_context_ok` only means the current public venue context is not obviously blocking a small repeat observation. It does not cover account fees, maker queue, liquidation buffer, borrow, or operational risk.
