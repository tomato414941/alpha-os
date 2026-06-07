# Current Funding Dislocation Monitor

This repeats the current dislocation watchlist over a short window. It is a persistence check, not a trade instruction.

| source | action | asset | long | short | obs | mean edge | min edge | mean net 8h | mean net 24h | positive net24 | liquidity | friction |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| okx_hl_current | paper_24h_monitor | STABLE | OkxSwap | HlPerp | 3 | 2.008195 | 1.997650 | -0.001565 | 0.002103 | 1.000000 | 11833.20 | 0.003399 |
| predicted_cross_venue | current_funding_monitor | STABLE | BybitPerp | HlPerp | 3 | 2.475677 | 2.471074 |  |  |  | 1183314.74 | 0.003300 |
| predicted_cross_venue | current_funding_monitor | SAGA | HlPerp | BybitPerp | 3 | 1.454044 | 1.447843 |  |  |  | 186344.40 | 0.002428 |
| hl_single_venue | current_funding_monitor | SAGA | HlPerp | cash_or_spot_proxy | 3 | 1.331077 | 1.326578 |  |  |  | 186355.19 | 0.002914 |
| predicted_cross_venue | current_funding_monitor | kNEIRO | HlPerp | BybitPerp | 3 | 1.235593 | 1.235062 |  |  |  | 294160.90 | 0.001451 |
| hl_single_venue | current_funding_monitor | kNEIRO | HlPerp | cash_or_spot_proxy | 3 | 1.116639 | 1.114401 |  |  |  | 294160.90 | 0.001243 |
| hl_single_venue | current_funding_monitor | SNX | HlPerp | cash_or_spot_proxy | 3 | 1.111456 | 1.104971 |  |  |  | 249083.18 | 0.002233 |
| predicted_cross_venue | current_funding_monitor | SNX | HlPerp | BinPerp | 3 | 1.101985 | 1.097773 |  |  |  | 249031.52 | 0.002385 |
| predicted_cross_venue | current_funding_monitor | AIXBT | HlPerp | BinPerp | 3 | 1.098454 | 1.093315 |  |  |  | 249251.65 | 0.002281 |
| predicted_cross_venue | current_funding_monitor | WLD | BinPerp | HlPerp | 3 | 1.034731 | 1.034731 |  |  |  | 48667897.00 | 0.000923 |
| hl_single_venue | current_funding_monitor | AIXBT | HlPerp | cash_or_spot_proxy | 3 | 0.977530 | 0.976999 |  |  |  | 249324.49 | 0.002325 |
| predicted_cross_venue | current_funding_monitor | BSV | HlPerp | BybitPerp | 3 | 0.961272 | 0.957466 |  |  |  | 203909.92 | 0.004411 |
| hl_single_venue | current_funding_monitor | ZEC | HlPerp | cash_or_spot_proxy | 3 | 0.914070 | 0.911113 |  |  |  | 320602182.46 | 0.000685 |
| hl_single_venue | current_funding_monitor | MON | HlPerp | cash_or_spot_proxy | 3 | 0.839156 | 0.837059 |  |  |  | 2408724.73 | 0.001711 |
| hl_single_venue | current_funding_monitor | BSV | HlPerp | cash_or_spot_proxy | 3 | 0.831142 | 0.824764 |  |  |  | 203909.92 | 0.004401 |
| hl_single_venue | current_funding_monitor | MORPHO | HlPerp | cash_or_spot_proxy | 3 | 0.764213 | 0.760612 |  |  |  | 794635.40 | 0.002049 |
| hl_single_venue | current_funding_monitor | STABLE | HlPerp | cash_or_spot_proxy | 3 | 0.684147 | 0.681041 |  |  |  | 1183319.76 | 0.003250 |
| hl_single_venue | current_funding_monitor | BIO | HlPerp | cash_or_spot_proxy | 3 | 0.631559 | 0.630601 |  |  |  | 631007.09 | 0.001502 |
| hl_single_venue | current_funding_monitor | BABY | HlPerp | cash_or_spot_proxy | 3 | 0.624870 | 0.620524 |  |  |  | 1739229.68 | 0.002489 |
| hl_single_venue | current_funding_monitor | POPCAT | HlPerp | cash_or_spot_proxy | 3 | 0.619325 | 0.617965 |  |  |  | 460995.60 | 0.002005 |
| predicted_cross_venue | current_funding_monitor | PROVE | HlPerp | BybitPerp | 3 | 0.614700 | 0.612049 |  |  |  | 136039.44 | 0.002251 |
| hl_single_venue | current_funding_monitor | NIL | HlPerp | cash_or_spot_proxy | 3 | 0.613187 | 0.610684 |  |  |  | 515463.34 | 0.004362 |
| predicted_cross_venue | current_funding_monitor | MORPHO | HlPerp | BinPerp | 3 | 0.605993 | 0.603763 |  |  |  | 794352.13 | 0.001716 |
| hl_single_venue | current_funding_monitor | SEI | HlPerp | cash_or_spot_proxy | 3 | 0.598631 | 0.597462 |  |  |  | 913321.73 | 0.001303 |
| okx_hl_current | paper_24h_monitor | WLD | OkxSwap | HlPerp | 2 | 0.872362 | 0.870058 | -0.000313 | 0.001280 | 1.000000 | 486669.83 | 0.001110 |

## Interpretation

Rows that appear in every sample with positive 24-hour proxy are the next monitor candidates. Rows without an executable hedge stay as funding alerts, not paper-trade candidates.
