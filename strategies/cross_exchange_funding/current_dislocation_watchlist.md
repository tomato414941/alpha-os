# Current Funding Dislocation Watchlist

This is a current-state watchlist, not a backtest and not a trade instruction. It combines current Hyperliquid funding, predicted cross-venue funding spreads, and OKX-Hyperliquid rough execution proxies.

| source | action | asset | long | short | annualized edge | net 8h | net 24h | liquidity | friction | reason |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| predicted_cross_venue | current_funding_monitor | STABLE | BybitPerp | HlPerp | 2.514005 |  |  | 1176450.45 | 0.002986 | Hyperliquid context available |
| okx_hl_current | paper_24h_monitor | STABLE | OkxSwap | HlPerp | 1.986599 | -0.001856 | 0.001772 | 11763.49 | 0.003670 | OKX and Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | SAGA | HlPerp | BybitPerp | 1.535372 |  |  | 186383.69 | 0.002198 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | SAGA | HlPerp | cash_or_spot_proxy | 1.432997 |  |  | 186295.31 | 0.002204 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | kNEIRO | HlPerp | BybitPerp | 1.232342 |  |  | 298296.38 | 0.001407 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | SNX | HlPerp | cash_or_spot_proxy | 1.185155 |  |  | 250853.53 | 0.002377 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | AIXBT | HlPerp | BinPerp | 1.161803 |  |  | 248936.78 | 0.002410 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | SNX | HlPerp | BinPerp | 1.140710 |  |  | 250897.16 | 0.002371 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | kNEIRO | HlPerp | cash_or_spot_proxy | 1.127829 |  |  | 298252.94 | 0.001567 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | BSV | HlPerp | BybitPerp | 1.075687 |  |  | 205284.92 | 0.004041 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | AIXBT | HlPerp | cash_or_spot_proxy | 1.063877 |  |  | 248535.63 | 0.002087 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | WLD | BinPerp | HlPerp | 1.025001 |  |  | 49129460.54 | 0.000820 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | BSV | HlPerp | cash_or_spot_proxy | 0.964942 |  |  | 205265.39 | 0.003267 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | ZEC | HlPerp | cash_or_spot_proxy | 0.922833 |  |  | 328501311.64 | 0.000460 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | MON | HlPerp | cash_or_spot_proxy | 0.881595 |  |  | 2415730.54 | 0.001565 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | NIL | HlPerp | cash_or_spot_proxy | 0.755754 |  |  | 516017.85 | 0.002770 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | STABLE | HlPerp | cash_or_spot_proxy | 0.745648 |  |  | 1176331.44 | 0.002781 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | MORPHO | HlPerp | cash_or_spot_proxy | 0.738904 |  |  | 800502.38 | 0.001654 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | BABY | BybitPerp | HlPerp | 0.724325 |  |  | 1749689.58 | 0.002526 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | S | HlPerp | cash_or_spot_proxy | 0.706561 |  |  | 149385.69 | 0.001522 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | GMT | BybitPerp | HlPerp | 0.700948 |  |  | 106584.93 | 0.003958 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | PROVE | HlPerp | BybitPerp | 0.658968 |  |  | 136279.94 | 0.002070 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | SEI | HlPerp | cash_or_spot_proxy | 0.649450 |  |  | 921192.41 | 0.001320 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | 0G | HlPerp | BybitPerp | 0.633744 |  |  | 162565.15 | 0.001457 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | BIO | HlPerp | cash_or_spot_proxy | 0.625974 |  |  | 630608.89 | 0.001488 | long_perp_receives_funding |

## Interpretation

`paper_24h_monitor` means the current rough 24-hour proxy is positive, but it still needs real fee tier, fill, margin, and borrow/collateral checks. `current_funding_monitor` means the funding rate is large enough to watch, but no executable hedge has been proven.
