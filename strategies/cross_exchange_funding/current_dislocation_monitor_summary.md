# Current Funding Dislocation Monitor

This repeats the current dislocation watchlist over a short window. It is a persistence check, not a trade instruction.

| source | action | asset | long | short | obs | mean edge | min edge | mean net 8h | mean net 24h | positive net24 | liquidity | friction |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hl_single_venue | current_funding_monitor | LAYER | HlPerp | cash_or_spot_proxy | 3 | 11.358028 | 11.347884 |  |  |  | 2165013.10 | 0.002736 |
| hl_single_venue | current_funding_monitor | STABLE | HlPerp | cash_or_spot_proxy | 3 | 3.178984 | 3.167197 |  |  |  | 461685.32 | 0.001127 |
| predicted_cross_venue | current_funding_monitor | STABLE | BybitPerp | HlPerp | 3 | 2.397066 | 2.369280 |  |  |  | 461685.32 | 0.000902 |
| hl_single_venue | current_funding_monitor | ZEC | HlPerp | cash_or_spot_proxy | 3 | 1.938589 | 1.935999 |  |  |  | 312699009.98 | 0.000434 |
| hl_single_venue | current_funding_monitor | MOVE | HlPerp | cash_or_spot_proxy | 3 | 1.247863 | 1.237061 |  |  |  | 1622247.39 | 0.002650 |
| predicted_cross_venue | current_funding_monitor | ENA | BybitPerp | HlPerp | 3 | 1.074602 | 1.027324 |  |  |  | 9681698.45 | 0.000914 |
| predicted_cross_venue | current_funding_monitor | AERO | BybitPerp | HlPerp | 3 | 1.012245 | 0.986243 |  |  |  | 669215.39 | 0.002689 |
| predicted_cross_venue | current_funding_monitor | NIL | BybitPerp | HlPerp | 3 | 0.988718 | 0.942608 |  |  |  | 543933.01 | 0.003359 |
| predicted_cross_venue | current_funding_monitor | ALT | BybitPerp | HlPerp | 3 | 0.938605 | 0.882198 |  |  |  | 182670.72 | 0.003932 |
| hl_single_venue | current_funding_monitor | AERO | HlPerp | cash_or_spot_proxy | 3 | 0.904338 | 0.903682 |  |  |  | 669215.39 | 0.002654 |
| predicted_cross_venue | current_funding_monitor | ZEC | HlPerp | BinPerp | 3 | 0.897278 | 0.888914 |  |  |  | 312691542.75 | 0.000457 |
| predicted_cross_venue | current_funding_monitor | GMX | BybitPerp | HlPerp | 3 | 0.886892 | 0.857462 |  |  |  | 182426.00 | 0.003857 |
| predicted_cross_venue | current_funding_monitor | BABY | BybitPerp | HlPerp | 3 | 0.834877 | 0.830749 |  |  |  | 831387.13 | 0.002922 |
| hl_single_venue | current_funding_monitor | AVNT | HlPerp | cash_or_spot_proxy | 3 | 0.833503 | 0.832578 |  |  |  | 482897.35 | 0.001770 |
| predicted_cross_venue | current_funding_monitor | kBONK | BybitPerp | HlPerp | 3 | 0.797708 | 0.782815 |  |  |  | 759116.38 | 0.000924 |
| predicted_cross_venue | current_funding_monitor | AXS | BybitPerp | HlPerp | 3 | 0.787392 | 0.760385 |  |  |  | 124942.88 | 0.003332 |
| hl_single_venue | current_funding_monitor | BABY | HlPerp | cash_or_spot_proxy | 3 | 0.752886 | 0.748637 |  |  |  | 831436.04 | 0.002518 |
| hl_single_venue | current_funding_monitor | S | HlPerp | cash_or_spot_proxy | 3 | 0.740869 | 0.738353 |  |  |  | 205843.15 | 0.002178 |
| predicted_cross_venue | current_funding_monitor | AZTEC | BybitPerp | HlPerp | 3 | 0.668556 | 0.609324 |  |  |  | 309425.82 | 0.003663 |
| predicted_cross_venue | current_funding_monitor | kFLOKI | BybitPerp | HlPerp | 3 | 0.664534 | 0.637126 |  |  |  | 114058.59 | 0.001147 |
| predicted_cross_venue | current_funding_monitor | SAGA | HlPerp | BybitPerp | 3 | 0.651868 | 0.651775 |  |  |  | 182494.01 | 0.002863 |
| predicted_cross_venue | current_funding_monitor | LAYER | BinPerp | HlPerp | 3 | 0.631916 | 0.619498 |  |  |  | 2164536.83 | 0.002754 |
| hl_single_venue | current_funding_monitor | IO | HlPerp | cash_or_spot_proxy | 2 | 0.604158 | 0.603989 |  |  |  | 780317.10 | 0.002183 |
| okx_hl_current | paper_24h_monitor | LAYER | OkxSwap | HlPerp | 1 | 1.025298 | 1.025298 | -0.001863 | 0.000009 | 1.000000 | 21642.49 | 0.002800 |
| predicted_cross_venue | current_funding_monitor | ME | BybitPerp | HlPerp | 1 | 0.903594 | 0.903594 |  |  |  | 364832.05 | 0.004685 |

## Interpretation

Rows that appear in every sample with positive 24-hour proxy are the next monitor candidates. Rows without an executable hedge stay as funding alerts, not paper-trade candidates.
