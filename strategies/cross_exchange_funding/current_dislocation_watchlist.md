# Current Funding Dislocation Watchlist

This is a current-state watchlist, not a backtest and not a trade instruction. It combines current Hyperliquid funding, predicted cross-venue funding spreads, and OKX-Hyperliquid rough execution proxies.

| source | action | asset | long | short | annualized edge | net 8h | net 24h | liquidity | friction | reason |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hl_single_venue | current_funding_monitor | MOVE | HlPerp | cash_or_spot_proxy | 7.953939 |  |  | 357189.74 | 0.004248 | long_perp_receives_funding |
| okx_hl_current | paper_24h_monitor | MOVE | OkxSwap | HlPerp | 3.289550 | -0.001165 | 0.004843 | 10608.09 | 0.004170 | OKX and Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | ME | BinPerp | HlPerp | 3.234323 |  |  | 277992.56 | 0.004805 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | STABLE | HlPerp | cash_or_spot_proxy | 2.763176 |  |  | 757479.65 | 0.002296 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | MOVE | BinPerp | HlPerp | 1.831169 |  |  | 1060808.97 | 0.003692 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | IO | HlPerp | cash_or_spot_proxy | 1.061861 |  |  | 514430.24 | 0.001591 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | BIO | BybitPerp | HlPerp | 0.909248 |  |  | 748004.33 | 0.002499 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | AIXBT | HlPerp | cash_or_spot_proxy | 0.757571 |  |  | 332868.06 | 0.002190 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | ZEC | HlPerp | cash_or_spot_proxy | 0.741535 |  |  | 439029767.71 | 0.000208 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | IP | HlPerp | cash_or_spot_proxy | 0.709831 |  |  | 524635.24 | 0.001297 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | SAND | HlPerp | cash_or_spot_proxy | 0.619573 |  |  | 213819.72 | 0.002822 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | ME | HlPerp | cash_or_spot_proxy | 0.614215 |  |  | 187325.38 | 0.001839 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | IMX | BinPerp | HlPerp | 0.586153 |  |  | 1057974.68 | 0.003307 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | COMP | BybitPerp | HlPerp | 0.571885 |  |  | 173305.42 | 0.003076 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | kNEIRO | HlPerp | cash_or_spot_proxy | 0.560123 |  |  | 259420.26 | 0.001088 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | W | HlPerp | cash_or_spot_proxy | 0.551368 |  |  | 220667.24 | 0.004167 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | kSHIB | HlPerp | cash_or_spot_proxy | 0.531050 |  |  | 820722.75 | 0.000847 | long_perp_receives_funding |
| okx_hl_current | paper_24h_monitor | HYPE | OkxSwap | HlPerp | 0.248565 | -0.000122 | 0.000332 | 4072267.81 | 0.000349 | OKX and Hyperliquid context available |
| okx_hl_current | blocked_by_cost_or_capacity | ME | OkxSwap | HlPerp | 3.318661 | -0.003054 | 0.003007 | 2779.93 | 0.006085 | wide Hyperliquid impact spread |
| predicted_cross_venue | thin_or_wide_watch | STABLE | BinPerp | BybitPerp | 1.377379 |  |  | 0.00 | 0.000000 | Hyperliquid not involved; external venue feasibility still unknown |
| predicted_cross_venue | thin_or_wide_watch | UMA | BybitPerp | BinPerp | 1.184352 |  |  | 0.00 | 0.000000 | Hyperliquid not involved; external venue feasibility still unknown |
| okx_hl_current | blocked_by_cost_or_capacity | STABLE | OkxSwap | HlPerp | 0.767234 | -0.002419 | -0.001018 | 7656.11 | 0.003120 | OKX and Hyperliquid context available |
| predicted_cross_venue | thin_or_wide_watch | MEME | BybitPerp | BinPerp | 0.655598 |  |  | 0.00 | 0.000000 | Hyperliquid not involved; external venue feasibility still unknown |
| okx_hl_current | blocked_by_cost_or_capacity | JUP | HlPerp | OkxSwap | 0.630573 | -0.001635 | -0.000483 | 14930.18 | 0.002211 | OKX and Hyperliquid context available |
| predicted_cross_venue | thin_or_wide_watch | MET | BybitPerp | BinPerp | 0.620602 |  |  | 0.00 | 0.000000 | Hyperliquid not involved; external venue feasibility still unknown |

## Interpretation

`paper_24h_monitor` means the current rough 24-hour proxy is positive, but it still needs real fee tier, fill, margin, and borrow/collateral checks. `current_funding_monitor` means the funding rate is large enough to watch, but no executable hedge has been proven.
