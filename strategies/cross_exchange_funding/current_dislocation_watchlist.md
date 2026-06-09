# Current Funding Dislocation Watchlist

This is a current-state watchlist, not a backtest and not a trade instruction. It combines current Hyperliquid funding, predicted cross-venue funding spreads, and OKX-Hyperliquid rough execution proxies.

| source | action | asset | long | short | annualized edge | net 8h | net 24h | liquidity | friction | reason |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hl_single_venue | current_funding_monitor | LAYER | HlPerp | cash_or_spot_proxy | 11.492001 |  |  | 2161247.32 | 0.002910 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | STABLE | HlPerp | cash_or_spot_proxy | 3.064576 |  |  | 458322.95 | 0.001560 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | STABLE | BybitPerp | HlPerp | 2.449270 |  |  | 462419.48 | 0.000799 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | ZEC | HlPerp | cash_or_spot_proxy | 1.994637 |  |  | 311760975.33 | 0.000885 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | ENA | BybitPerp | HlPerp | 1.166290 |  |  | 9816165.22 | 0.001080 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | MOVE | HlPerp | cash_or_spot_proxy | 1.157716 |  |  | 1619576.49 | 0.002616 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | ME | BybitPerp | HlPerp | 1.122689 |  |  | 374987.60 | 0.004682 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | NIL | BybitPerp | HlPerp | 1.080019 |  |  | 543977.66 | 0.002432 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | AERO | BybitPerp | HlPerp | 1.063992 |  |  | 669215.39 | 0.002588 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | ALT | BybitPerp | HlPerp | 1.051419 |  |  | 182689.28 | 0.002819 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | GMX | BybitPerp | HlPerp | 0.945751 |  |  | 182426.00 | 0.003714 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | ZEC | HlPerp | BinPerp | 0.888240 |  |  | 312727052.28 | 0.000407 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | AERO | HlPerp | cash_or_spot_proxy | 0.882164 |  |  | 667977.65 | 0.002865 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | AXS | BybitPerp | HlPerp | 0.842033 |  |  | 124976.61 | 0.003044 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | BABY | BybitPerp | HlPerp | 0.841218 |  |  | 831478.47 | 0.002021 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | kBONK | BybitPerp | HlPerp | 0.827492 |  |  | 759151.94 | 0.000925 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | S | HlPerp | cash_or_spot_proxy | 0.799606 |  |  | 200362.70 | 0.001973 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | AVNT | HlPerp | cash_or_spot_proxy | 0.788001 |  |  | 481290.66 | 0.002365 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | AZTEC | BybitPerp | HlPerp | 0.787020 |  |  | 309530.51 | 0.003315 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | BABY | HlPerp | cash_or_spot_proxy | 0.744652 |  |  | 829876.17 | 0.002283 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | kFLOKI | BybitPerp | HlPerp | 0.719349 |  |  | 114058.59 | 0.000923 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | PENGU | BybitPerp | HlPerp | 0.714553 |  |  | 1607882.20 | 0.000457 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | NEO | BybitPerp | HlPerp | 0.652073 |  |  | 104946.84 | 0.004179 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | POL | BybitPerp | HlPerp | 0.634277 |  |  | 458924.43 | 0.001732 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | VVV | BybitPerp | HlPerp | 0.623606 |  |  | 18728324.64 | 0.002198 | Hyperliquid context available |

## Interpretation

`paper_24h_monitor` means the current rough 24-hour proxy is positive, but it still needs real fee tier, fill, margin, and borrow/collateral checks. `current_funding_monitor` means the funding rate is large enough to watch, but no executable hedge has been proven.
