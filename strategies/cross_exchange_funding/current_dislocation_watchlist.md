# Current Funding Dislocation Watchlist

This is a current-state watchlist, not a backtest and not a trade instruction. It combines current Hyperliquid funding, predicted cross-venue funding spreads, and OKX-Hyperliquid rough execution proxies.

| source | action | asset | long | short | annualized edge | net 8h | net 24h | liquidity | friction | reason |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hl_single_venue | current_funding_monitor | LAYER | HlPerp | cash_or_spot_proxy | 18.393339 |  |  | 1266789.65 | 0.004598 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | MOVE | HlPerp | BybitPerp | 1.928521 |  |  | 1497579.27 | 0.001729 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | IO | HlPerp | cash_or_spot_proxy | 1.416071 |  |  | 680209.86 | 0.002479 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | STABLE | HlPerp | cash_or_spot_proxy | 1.405638 |  |  | 477041.81 | 0.002030 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | S | HlPerp | BinPerp | 1.255562 |  |  | 155905.00 | 0.001123 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | S | HlPerp | cash_or_spot_proxy | 1.243906 |  |  | 158975.00 | 0.001352 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | IO | HlPerp | BinPerp | 1.177008 |  |  | 678798.42 | 0.002339 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | AVNT | HlPerp | cash_or_spot_proxy | 1.134982 |  |  | 494746.63 | 0.001613 | long_perp_receives_funding |
| okx_hl_current | paper_24h_monitor | MOVE | HlPerp | OkxSwap | 1.125827 | -0.001421 | 0.000635 | 14975.79 | 0.002449 | OKX and Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | STABLE | BinPerp | HlPerp | 1.070465 |  |  | 471704.61 | 0.001912 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | ZEC | HlPerp | cash_or_spot_proxy | 1.020960 |  |  | 323392532.09 | 0.000422 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | MOVE | HlPerp | cash_or_spot_proxy | 1.016327 |  |  | 1498635.91 | 0.001727 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | AIXBT | HlPerp | BinPerp | 0.933777 |  |  | 250868.84 | 0.002645 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | AIXBT | HlPerp | cash_or_spot_proxy | 0.851603 |  |  | 256545.13 | 0.002511 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | AVNT | HlPerp | BinPerp | 0.844342 |  |  | 490440.29 | 0.001884 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | MELANIA | BinPerp | HlPerp | 0.772435 |  |  | 344869.50 | 0.001005 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | ATOM | HlPerp | cash_or_spot_proxy | 0.760028 |  |  | 345849.36 | 0.001952 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | LAYER | HlPerp | BinPerp | 0.686604 |  |  | 1265400.42 | 0.004844 | mark/oracle dislocation |
| predicted_cross_venue | current_funding_monitor | PROVE | HlPerp | BybitPerp | 0.669254 |  |  | 488289.56 | 0.002255 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | COMP | BybitPerp | HlPerp | 0.664050 |  |  | 247636.57 | 0.002013 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | PROVE | HlPerp | cash_or_spot_proxy | 0.558907 |  |  | 489396.10 | 0.002053 | long_perp_receives_funding |
| okx_hl_current | paper_24h_monitor | ZEC | HlPerp | OkxSwap | 0.368899 | -0.000266 | 0.000408 | 154394.61 | 0.000603 | OKX and Hyperliquid context available |
| okx_hl_current | paper_24h_monitor | AAVE | OkxSwap | HlPerp | 0.331259 | -0.000421 | 0.000184 | 47226.26 | 0.000724 | OKX and Hyperliquid context available |
| okx_hl_current | paper_24h_monitor | LINK | OkxSwap | HlPerp | 0.230123 | -0.000200 | 0.000220 | 48177.20 | 0.000411 | OKX and Hyperliquid context available |
| okx_hl_current | paper_24h_monitor | DOGE | OkxSwap | HlPerp | 0.128700 | -0.000170 | 0.000065 | 109019.78 | 0.000288 | OKX and Hyperliquid context available |

## Interpretation

`paper_24h_monitor` means the current rough 24-hour proxy is positive, but it still needs real fee tier, fill, margin, and borrow/collateral checks. `current_funding_monitor` means the funding rate is large enough to watch, but no executable hedge has been proven.
