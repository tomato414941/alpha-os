# Current Funding Dislocation Watchlist

This is a current-state watchlist, not a backtest and not a trade instruction. It combines current Hyperliquid funding, predicted cross-venue funding spreads, and OKX-Hyperliquid rough execution proxies.

| source | action | asset | long | short | annualized edge | net 8h | net 24h | liquidity | friction | reason |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hl_single_venue | current_funding_monitor | STABLE | HlPerp | cash_or_spot_proxy | 4.075735 |  |  | 761322.73 | 0.002298 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | LAYER | HlPerp | BinPerp | 3.906031 |  |  | 280966.72 | 0.004175 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | MOVE | HlPerp | cash_or_spot_proxy | 2.877765 |  |  | 1226734.20 | 0.002703 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | BIO | BybitPerp | HlPerp | 2.587792 |  |  | 1380612.25 | 0.001799 | Hyperliquid context available |
| okx_hl_current | paper_24h_monitor | BIO | OkxSwap | HlPerp | 2.101435 | -0.000438 | 0.003400 | 13806.12 | 0.002357 | OKX and Hyperliquid context available |
| okx_hl_current | paper_24h_monitor | MOVE | OkxSwap | HlPerp | 1.696075 | -0.001085 | 0.002012 | 12867.11 | 0.002634 | OKX and Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | PROVE | HlPerp | BybitPerp | 1.165873 |  |  | 491182.91 | 0.001684 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | ZEC | BybitPerp | HlPerp | 1.159189 |  |  | 350549960.23 | 0.000316 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | PROVE | HlPerp | cash_or_spot_proxy | 0.976362 |  |  | 494969.16 | 0.001690 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | ME | HlPerp | cash_or_spot_proxy | 0.953190 |  |  | 298745.14 | 0.002408 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | HYPER | HlPerp | BinPerp | 0.931749 |  |  | 233943.87 | 0.001727 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | ME | BinPerp | HlPerp | 0.885113 |  |  | 297072.75 | 0.001284 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | S | HlPerp | BybitPerp | 0.866776 |  |  | 195710.88 | 0.001592 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | BABY | HlPerp | BybitPerp | 0.812149 |  |  | 1050245.11 | 0.001811 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | ZEC | HlPerp | cash_or_spot_proxy | 0.778720 |  |  | 408603031.27 | 0.000492 | long_perp_receives_funding |
| okx_hl_current | paper_24h_monitor | MEGA | OkxSwap | HlPerp | 0.737392 | -0.000708 | 0.000638 | 24847.79 | 0.001382 | OKX and Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | AERO | BybitPerp | HlPerp | 0.616286 |  |  | 650033.51 | 0.002173 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | JUP | HlPerp | BybitPerp | 0.573170 |  |  | 1726597.98 | 0.001340 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | AIXBT | HlPerp | cash_or_spot_proxy | 0.569933 |  |  | 328899.01 | 0.001703 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | COMP | BybitPerp | HlPerp | 0.565852 |  |  | 174053.20 | 0.004215 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | OP | HlPerp | cash_or_spot_proxy | 0.561795 |  |  | 854447.15 | 0.001123 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | MINA | HlPerp | BybitPerp | 0.537121 |  |  | 135391.24 | 0.002246 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | TRUMP | HlPerp | cash_or_spot_proxy | 0.535037 |  |  | 1777004.84 | 0.000637 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | VIRTUAL | BybitPerp | HlPerp | 0.513739 |  |  | 1876814.80 | 0.000965 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | BABY | HlPerp | cash_or_spot_proxy | 0.505229 |  |  | 1057277.55 | 0.002202 | long_perp_receives_funding |

## Interpretation

`paper_24h_monitor` means the current rough 24-hour proxy is positive, but it still needs real fee tier, fill, margin, and borrow/collateral checks. `current_funding_monitor` means the funding rate is large enough to watch, but no executable hedge has been proven.
