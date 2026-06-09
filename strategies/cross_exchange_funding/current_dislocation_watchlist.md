# Current Funding Dislocation Watchlist

This is a current-state watchlist, not a backtest and not a trade instruction. It combines current Hyperliquid funding, predicted cross-venue funding spreads, and OKX-Hyperliquid rough execution proxies.

| source | action | asset | long | short | annualized edge | net 8h | net 24h | liquidity | friction | reason |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hl_single_venue | current_funding_monitor | LAYER | HlPerp | cash_or_spot_proxy | 14.105704 |  |  | 1985439.02 | 0.003076 | long_perp_receives_funding |
| okx_hl_current | paper_24h_monitor | HMSTR | HlPerp | OkxSwap | 3.098332 | -0.003712 | 0.001947 | 15122.03 | 0.006541 | wide Hyperliquid impact spread |
| predicted_cross_venue | current_funding_monitor | STABLE | BybitPerp | HlPerp | 2.660825 |  |  | 474374.61 | 0.001062 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | STABLE | HlPerp | cash_or_spot_proxy | 2.019238 |  |  | 454701.25 | 0.002890 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | ZEC | HlPerp | cash_or_spot_proxy | 1.809971 |  |  | 317968556.90 | 0.000692 | long_perp_receives_funding |
| okx_hl_current | paper_24h_monitor | LAYER | OkxSwap | HlPerp | 1.312367 | -0.002123 | 0.000274 | 21275.87 | 0.003321 | OKX and Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | AZTEC | BybitPerp | HlPerp | 1.303422 |  |  | 294717.58 | 0.002776 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | GMX | BybitPerp | HlPerp | 1.286581 |  |  | 184829.60 | 0.002228 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | BRETT | BybitPerp | HlPerp | 1.268703 |  |  | 182738.90 | 0.003038 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | SUSHI | BybitPerp | HlPerp | 1.156725 |  |  | 171366.92 | 0.001780 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | WCT | BybitPerp | HlPerp | 1.152553 |  |  | 129445.52 | 0.001624 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | S | HlPerp | cash_or_spot_proxy | 1.129738 |  |  | 167997.90 | 0.000931 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | S | HlPerp | BinPerp | 1.051128 |  |  | 186503.65 | 0.000737 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | SAGA | HlPerp | BinPerp | 1.036319 |  |  | 164530.44 | 0.002283 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | AERO | HlPerp | cash_or_spot_proxy | 1.025410 |  |  | 708499.22 | 0.000715 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | kBONK | BybitPerp | HlPerp | 0.958738 |  |  | 795165.62 | 0.000911 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | NEO | BybitPerp | HlPerp | 0.917139 |  |  | 101774.75 | 0.002028 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | NIL | BinPerp | HlPerp | 0.819848 |  |  | 582005.98 | 0.003109 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | MOVE | HlPerp | BybitPerp | 0.806334 |  |  | 1586265.36 | 0.002629 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | MNT | BybitPerp | HlPerp | 0.771066 |  |  | 246213.11 | 0.000822 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | NIL | HlPerp | cash_or_spot_proxy | 0.756048 |  |  | 601077.89 | 0.002581 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | APE | BybitPerp | HlPerp | 0.741238 |  |  | 1447689.16 | 0.001758 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | kNEIRO | HlPerp | cash_or_spot_proxy | 0.708234 |  |  | 437035.03 | 0.001739 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | VIRTUAL | HlPerp | cash_or_spot_proxy | 0.634270 |  |  | 1439071.91 | 0.000770 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | LAYER | BinPerp | HlPerp | 0.617926 |  |  | 2127587.06 | 0.002665 | mark/oracle dislocation |

## Interpretation

`paper_24h_monitor` means the current rough 24-hour proxy is positive, but it still needs real fee tier, fill, margin, and borrow/collateral checks. `current_funding_monitor` means the funding rate is large enough to watch, but no executable hedge has been proven.
