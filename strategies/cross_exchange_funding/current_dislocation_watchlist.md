# Current Funding Dislocation Watchlist

This is a current-state watchlist, not a backtest and not a trade instruction. It combines current Hyperliquid funding, predicted cross-venue funding spreads, and OKX-Hyperliquid rough execution proxies.

| source | action | asset | long | short | annualized edge | net 8h | net 24h | liquidity | friction | reason |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hl_single_venue | current_funding_monitor | MOVE | HlPerp | cash_or_spot_proxy | 7.953939 |  |  | 357189.74 | 0.004248 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | STABLE | HlPerp | cash_or_spot_proxy | 2.763176 |  |  | 757479.65 | 0.002296 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | ZORA | BybitPerp | HlPerp | 1.889247 |  |  | 120877.16 | 0.003098 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | AERO | BybitPerp | HlPerp | 1.671080 |  |  | 687774.85 | 0.001960 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | JUP | BybitPerp | HlPerp | 1.605780 |  |  | 1405843.77 | 0.001650 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | IO | HlPerp | cash_or_spot_proxy | 1.061861 |  |  | 514430.24 | 0.001591 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | PNUT | BybitPerp | HlPerp | 1.025882 |  |  | 275249.64 | 0.001435 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | IO | BybitPerp | HlPerp | 0.980649 |  |  | 582704.83 | 0.001896 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | POPCAT | BybitPerp | HlPerp | 0.968941 |  |  | 363454.67 | 0.001146 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | BERA | BybitPerp | HlPerp | 0.968732 |  |  | 472839.85 | 0.002044 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | ONDO | BybitPerp | HlPerp | 0.925231 |  |  | 13497430.21 | 0.001426 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | VIRTUAL | BybitPerp | HlPerp | 0.915386 |  |  | 1698539.91 | 0.001201 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | TRUMP | BybitPerp | HlPerp | 0.895904 |  |  | 1715053.81 | 0.000834 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | PENGU | BybitPerp | HlPerp | 0.848800 |  |  | 2590974.03 | 0.000731 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | RENDER | BybitPerp | HlPerp | 0.845931 |  |  | 1119269.91 | 0.002416 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | AIXBT | HlPerp | cash_or_spot_proxy | 0.757571 |  |  | 332868.06 | 0.002190 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | ZEC | HlPerp | cash_or_spot_proxy | 0.741535 |  |  | 439029767.71 | 0.000208 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | IP | HlPerp | cash_or_spot_proxy | 0.709831 |  |  | 524635.24 | 0.001297 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | IMX | BinPerp | HlPerp | 0.703559 |  |  | 1056294.82 | 0.002564 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | kBONK | BybitPerp | HlPerp | 0.662935 |  |  | 1460041.38 | 0.000905 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | SAND | HlPerp | cash_or_spot_proxy | 0.619573 |  |  | 213819.72 | 0.002822 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | XPL | BybitPerp | HlPerp | 0.616835 |  |  | 8373843.90 | 0.001617 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | ME | HlPerp | cash_or_spot_proxy | 0.614215 |  |  | 187325.38 | 0.001839 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | kNEIRO | HlPerp | cash_or_spot_proxy | 0.560123 |  |  | 259420.26 | 0.001088 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | W | HlPerp | cash_or_spot_proxy | 0.551368 |  |  | 220667.24 | 0.004167 | long_perp_receives_funding |

## Interpretation

`paper_24h_monitor` means the current rough 24-hour proxy is positive, but it still needs real fee tier, fill, margin, and borrow/collateral checks. `current_funding_monitor` means the funding rate is large enough to watch, but no executable hedge has been proven.
