# Current Funding Dislocation Watchlist

This is a current-state watchlist, not a backtest and not a trade instruction. It combines current Hyperliquid funding, predicted cross-venue funding spreads, and OKX-Hyperliquid rough execution proxies.

| source | action | asset | long | short | annualized edge | net 8h | net 24h | liquidity | friction | reason |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hl_single_venue | current_funding_monitor | STABLE | HlPerp | cash_or_spot_proxy | 4.010231 |  |  | 880532.05 | 0.002790 | long_perp_receives_funding |
| okx_hl_current | paper_24h_monitor | MON | OkxSwap | HlPerp | 1.123778 | -0.000990 | 0.001062 | 41147.74 | 0.002017 | OKX and Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | MON | BinPerp | HlPerp | 0.944340 |  |  | 4114773.94 | 0.001388 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | XMR | cash_or_spot_proxy | HlPerp | 0.853000 |  |  | 8914215.22 | 0.000789 | short_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | ZEC | BinPerp | HlPerp | 0.847316 |  |  | 509693652.20 | 0.000364 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | ZK | HlPerp | BinPerp | 0.797871 |  |  | 324185.34 | 0.002300 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | BABY | HlPerp | BybitPerp | 0.785853 |  |  | 870100.45 | 0.002951 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | TAO | BybitPerp | HlPerp | 0.738275 |  |  | 8373026.88 | 0.000666 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | BABY | HlPerp | cash_or_spot_proxy | 0.732869 |  |  | 857533.13 | 0.003124 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | XMR | BinPerp | HlPerp | 0.729631 |  |  | 8951731.07 | 0.000398 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | AERO | HlPerp | cash_or_spot_proxy | 0.720909 |  |  | 558934.36 | 0.001972 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | LDO | BybitPerp | HlPerp | 0.655018 |  |  | 607458.24 | 0.000989 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | ZK | HlPerp | cash_or_spot_proxy | 0.649333 |  |  | 323597.81 | 0.002701 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | AERO | HlPerp | BinPerp | 0.639219 |  |  | 574861.25 | 0.001768 | Hyperliquid context available |
| okx_hl_current | paper_24h_monitor | ZEC | OkxSwap | HlPerp | 0.638064 | -0.000297 | 0.000868 | 101012.30 | 0.000880 | OKX and Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | RENDER | BybitPerp | HlPerp | 0.619485 |  |  | 948814.89 | 0.001302 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | MERL | HlPerp | BybitPerp | 0.607129 |  |  | 184449.06 | 0.003503 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | SAGA | HlPerp | BybitPerp | 0.603051 |  |  | 166283.79 | 0.002216 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | AIXBT | HlPerp | BinPerp | 0.594125 |  |  | 369659.66 | 0.002354 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | MERL | HlPerp | cash_or_spot_proxy | 0.589172 |  |  | 182789.41 | 0.002062 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | JTO | HlPerp | BybitPerp | 0.573863 |  |  | 9487612.53 | 0.001923 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | HYPER | BybitPerp | HlPerp | 0.569203 |  |  | 200679.44 | 0.003166 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | kNEIRO | HlPerp | BybitPerp | 0.567486 |  |  | 280941.11 | 0.001074 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | TRUMP | HlPerp | cash_or_spot_proxy | 0.554528 |  |  | 1302934.59 | 0.000889 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | SPX | HlPerp | BybitPerp | 0.546985 |  |  | 1466240.98 | 0.001972 | Hyperliquid context available |

## Interpretation

`paper_24h_monitor` means the current rough 24-hour proxy is positive, but it still needs real fee tier, fill, margin, and borrow/collateral checks. `current_funding_monitor` means the funding rate is large enough to watch, but no executable hedge has been proven.
