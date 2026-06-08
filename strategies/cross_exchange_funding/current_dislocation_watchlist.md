# Current Funding Dislocation Watchlist

This is a current-state watchlist, not a backtest and not a trade instruction. It combines current Hyperliquid funding, predicted cross-venue funding spreads, and OKX-Hyperliquid rough execution proxies.

| source | action | asset | long | short | annualized edge | net 8h | net 24h | liquidity | friction | reason |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hl_single_venue | current_funding_monitor | STABLE | HlPerp | cash_or_spot_proxy | 5.717735 |  |  | 791373.12 | 0.002415 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | LAYER | HlPerp | cash_or_spot_proxy | 5.169090 |  |  | 290957.95 | 0.003907 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | LAYER | HlPerp | BinPerp | 3.824949 |  |  | 291004.59 | 0.002382 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | BIO | BybitPerp | HlPerp | 2.193771 |  |  | 1383760.22 | 0.001690 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | MOVE | HlPerp | BybitPerp | 1.851845 |  |  | 1310125.57 | 0.002063 | Hyperliquid context available |
| okx_hl_current | paper_24h_monitor | BIO | OkxSwap | HlPerp | 1.827139 | -0.000147 | 0.003191 | 13837.75 | 0.001815 | OKX and Hyperliquid context available |
| hl_single_venue | current_funding_monitor | MOVE | HlPerp | cash_or_spot_proxy | 1.735768 |  |  | 1302468.44 | 0.003206 | long_perp_receives_funding |
| okx_hl_current | paper_24h_monitor | MOVE | OkxSwap | HlPerp | 1.203145 | -0.001888 | 0.000309 | 13101.26 | 0.002987 | OKX and Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | PROVE | HlPerp | BybitPerp | 1.010500 |  |  | 491283.58 | 0.002177 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | BIO | HlPerp | cash_or_spot_proxy | 0.970757 |  |  | 1383562.56 | 0.001716 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | HYPER | HlPerp | BinPerp | 0.963137 |  |  | 233822.77 | 0.001900 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | PROVE | HlPerp | cash_or_spot_proxy | 0.917752 |  |  | 491268.29 | 0.002147 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | S | HlPerp | BybitPerp | 0.913977 |  |  | 194692.93 | 0.001587 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | HYPER | HlPerp | cash_or_spot_proxy | 0.843245 |  |  | 233822.77 | 0.001941 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | S | HlPerp | cash_or_spot_proxy | 0.795462 |  |  | 194692.93 | 0.001720 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | WCT | BybitPerp | HlPerp | 0.777735 |  |  | 128559.85 | 0.004881 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | ZEC | BinPerp | HlPerp | 0.764941 |  |  | 343704231.26 | 0.000127 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | AIXBT | HlPerp | BinPerp | 0.739405 |  |  | 308584.11 | 0.001654 | Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | AERO | BybitPerp | HlPerp | 0.728635 |  |  | 656547.12 | 0.002736 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | TRUMP | HlPerp | cash_or_spot_proxy | 0.651636 |  |  | 1787732.64 | 0.000559 | long_perp_receives_funding |
| hl_single_venue | current_funding_monitor | AIXBT | HlPerp | cash_or_spot_proxy | 0.630674 |  |  | 308565.27 | 0.001654 | long_perp_receives_funding |
| predicted_cross_venue | current_funding_monitor | IO | HlPerp | BybitPerp | 0.591492 |  |  | 539062.48 | 0.002094 | Hyperliquid context available |
| hl_single_venue | current_funding_monitor | INJ | HlPerp | cash_or_spot_proxy | 0.537202 |  |  | 4053930.39 | 0.000984 | long_perp_receives_funding |
| okx_hl_current | paper_24h_monitor | WLD | OkxSwap | HlPerp | 0.519196 | -0.000789 | 0.000159 | 382763.02 | 0.001263 | OKX and Hyperliquid context available |
| predicted_cross_venue | current_funding_monitor | 2Z | HlPerp | BybitPerp | 0.512503 |  |  | 388448.62 | 0.003353 | Hyperliquid context available |

## Interpretation

`paper_24h_monitor` means the current rough 24-hour proxy is positive, but it still needs real fee tier, fill, margin, and borrow/collateral checks. `current_funding_monitor` means the funding rate is large enough to watch, but no executable hedge has been proven.
