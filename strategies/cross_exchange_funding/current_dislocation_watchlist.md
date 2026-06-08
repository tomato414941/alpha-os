# Current Funding Dislocation Watchlist

This is a current-state watchlist, not a backtest and not a trade instruction. It combines current Hyperliquid funding, predicted cross-venue funding spreads, and OKX-Hyperliquid rough execution proxies.

| source | action | asset | long | short | annualized edge | net 8h | net 24h | liquidity | friction | reason |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hl_single_venue | current_funding_monitor | MOVE | HlPerp | cash_or_spot_proxy | 7.953939 |  |  | 357189.74 | 0.004248 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | MOVE | BybitPerp | HlPerp | 3.908892 |  |  | 1032943.29 | 0.004737 | mark/oracle dislocation |
| okx_hl_current | paper_24h_monitor | MOVE | OkxSwap | HlPerp | 3.135931 | -0.000927 | 0.004801 | 10329.43 | 0.003791 | OKX and Hyperliquid context available |
| hl_single_venue | current_funding_monitor | STABLE | HlPerp | cash_or_spot_proxy | 2.763176 |  |  | 757479.65 | 0.002296 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | IO | HlPerp | cash_or_spot_proxy | 1.061861 |  |  | 514430.24 | 0.001591 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | TRUMP | BybitPerp | HlPerp | 0.808256 |  |  | 1736805.13 | 0.000977 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | AIXBT | HlPerp | cash_or_spot_proxy | 0.757571 |  |  | 332868.06 | 0.002190 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | ZEC | HlPerp | cash_or_spot_proxy | 0.741535 |  |  | 439029767.71 | 0.000208 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | IP | HlPerp | cash_or_spot_proxy | 0.709831 |  |  | 524635.24 | 0.001297 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | BERA | BybitPerp | HlPerp | 0.683915 |  |  | 485313.80 | 0.002472 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | IMX | BinPerp | HlPerp | 0.661227 |  |  | 1065255.16 | 0.003925 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | VIRTUAL | BybitPerp | HlPerp | 0.655203 |  |  | 1729414.27 | 0.000859 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | PNUT | BybitPerp | HlPerp | 0.646089 |  |  | 279064.25 | 0.002623 | Hyperliquid context available |
| okx_hl_current | paper_24h_monitor | MEGA | OkxSwap | HlPerp | 0.631770 | -0.001013 | 0.000141 | 22969.85 | 0.001590 | OKX and Hyperliquid context available |
| hl_single_venue | current_funding_monitor | SAND | HlPerp | cash_or_spot_proxy | 0.619573 |  |  | 213819.72 | 0.002822 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | RENDER | BybitPerp | HlPerp | 0.619354 |  |  | 1140674.51 | 0.002651 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | ME | HlPerp | cash_or_spot_proxy | 0.614215 |  |  | 187325.38 | 0.001839 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | ZORA | BybitPerp | HlPerp | 0.609565 |  |  | 122215.50 | 0.002750 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | kNEIRO | HlPerp | cash_or_spot_proxy | 0.560123 |  |  | 259420.26 | 0.001088 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | W | HlPerp | cash_or_spot_proxy | 0.551368 |  |  | 220667.24 | 0.004167 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | kSHIB | HlPerp | cash_or_spot_proxy | 0.531050 |  |  | 820722.75 | 0.000847 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | NEAR | BybitPerp | HlPerp | 0.507423 |  |  | 74439618.01 | 0.000161 | Hyperliquid context available |
| okx_hl_current | paper_24h_monitor | TRX | OkxSwap | HlPerp | 0.200041 | -0.000319 | 0.000046 | 17228.20 | 0.000502 | OKX and Hyperliquid context available |
| okx_hl_current | paper_24h_monitor | ZEC | HlPerp | OkxSwap | 0.142184 | -0.000029 | 0.000231 | 107389.09 | 0.000159 | OKX and Hyperliquid context available |
| okx_hl_current | paper_24h_monitor | BTC | HlPerp | OkxSwap | 0.008336 | -0.000010 | 0.000005 | 376380.96 | 0.000018 | OKX and Hyperliquid context available |

## Interpretation

`paper_24h_monitor` means the current rough 24-hour proxy is positive, but it still needs real fee tier, fill, margin, and borrow/collateral checks. `current_funding_monitor` means the funding rate is large enough to watch, but no executable hedge has been proven.
