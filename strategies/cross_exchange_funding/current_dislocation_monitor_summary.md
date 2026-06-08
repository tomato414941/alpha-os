# Current Funding Dislocation Monitor

This repeats the current dislocation watchlist over a short window. It is a persistence check, not a trade instruction.

| source | action | asset | long | short | obs | mean edge | min edge | mean net 8h | mean net 24h | positive net24 | liquidity | friction |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| okx_hl_current | paper_24h_monitor | MON | OkxSwap | HlPerp | 4 | 1.098472 | 1.097628 | -0.000981 | 0.001026 | 1.000000 | 41030.88 | 0.001984 |
| okx_hl_current | paper_24h_monitor | JTO | HlPerp | OkxSwap | 4 | 0.552843 | 0.550433 | -0.000807 | 0.000203 | 1.000000 | 70776.92 | 0.001312 |
| hl_single_venue | current_funding_monitor | STABLE | HlPerp | cash_or_spot_proxy | 4 | 3.999175 | 3.985868 |  |  |  | 880975.74 | 0.002812 |
| predicted_cross_venue | current_funding_monitor | MON | BinPerp | HlPerp | 4 | 0.933788 | 0.932561 |  |  |  | 4103083.26 | 0.001412 |
| hl_single_venue | current_funding_monitor | XMR | cash_or_spot_proxy | HlPerp | 4 | 0.849258 | 0.848172 |  |  |  | 8941229.92 | 0.000891 |
| predicted_cross_venue | current_funding_monitor | BABY | HlPerp | BybitPerp | 4 | 0.809907 | 0.806489 |  |  |  | 859013.27 | 0.002770 |
| predicted_cross_venue | current_funding_monitor | ZEC | BinPerp | HlPerp | 4 | 0.782647 | 0.779542 |  |  |  | 508467465.70 | 0.000534 |
| hl_single_venue | current_funding_monitor | AERO | HlPerp | cash_or_spot_proxy | 4 | 0.762218 | 0.758157 |  |  |  | 564945.92 | 0.001628 |
| predicted_cross_venue | current_funding_monitor | ZK | HlPerp | BinPerp | 4 | 0.752900 | 0.750569 |  |  |  | 323821.52 | 0.002230 |
| predicted_cross_venue | current_funding_monitor | RENDER | BybitPerp | HlPerp | 4 | 0.750434 | 0.742088 |  |  |  | 944444.94 | 0.001461 |
| predicted_cross_venue | current_funding_monitor | XMR | BinPerp | HlPerp | 4 | 0.739946 | 0.738031 |  |  |  | 8939952.03 | 0.000828 |
| predicted_cross_venue | current_funding_monitor | TAO | BybitPerp | HlPerp | 4 | 0.710466 | 0.697116 |  |  |  | 8352958.51 | 0.000562 |
| hl_single_venue | current_funding_monitor | BABY | HlPerp | cash_or_spot_proxy | 4 | 0.700630 | 0.695505 |  |  |  | 859085.84 | 0.002680 |
| predicted_cross_venue | current_funding_monitor | MERL | HlPerp | BinPerp | 4 | 0.658941 | 0.648540 |  |  |  | 184118.54 | 0.003519 |
| predicted_cross_venue | current_funding_monitor | LDO | BybitPerp | HlPerp | 4 | 0.656543 | 0.655872 |  |  |  | 605340.87 | 0.000843 |
| hl_single_venue | current_funding_monitor | ZK | HlPerp | cash_or_spot_proxy | 4 | 0.649609 | 0.647786 |  |  |  | 323821.52 | 0.002326 |
| predicted_cross_venue | current_funding_monitor | APE | BybitPerp | HlPerp | 4 | 0.629866 | 0.575565 |  |  |  | 360002.71 | 0.001953 |
| predicted_cross_venue | current_funding_monitor | AERO | HlPerp | BinPerp | 4 | 0.601512 | 0.594055 |  |  |  | 564943.41 | 0.001631 |
| predicted_cross_venue | current_funding_monitor | TRUMP | HlPerp | BybitPerp | 4 | 0.590782 | 0.576694 |  |  |  | 1305264.56 | 0.000749 |
| predicted_cross_venue | current_funding_monitor | AIXBT | HlPerp | BinPerp | 4 | 0.582082 | 0.580304 |  |  |  | 367705.32 | 0.002074 |
| predicted_cross_venue | current_funding_monitor | SAGA | HlPerp | BybitPerp | 4 | 0.578211 | 0.573330 |  |  |  | 165885.71 | 0.002034 |
| predicted_cross_venue | current_funding_monitor | JTO | HlPerp | BybitPerp | 4 | 0.576580 | 0.571410 |  |  |  | 9472061.65 | 0.001137 |
| predicted_cross_venue | current_funding_monitor | NEO | BinPerp | HlPerp | 4 | 0.545458 | 0.544149 |  |  |  | 109122.48 | 0.001818 |
| hl_single_venue | current_funding_monitor | TRUMP | HlPerp | cash_or_spot_proxy | 4 | 0.543745 | 0.540001 |  |  |  | 1305277.05 | 0.000729 |
| hl_single_venue | current_funding_monitor | MERL | HlPerp | cash_or_spot_proxy | 3 | 0.533128 | 0.526211 |  |  |  | 184231.77 | 0.003570 |

## Interpretation

Rows that appear in every sample with positive 24-hour proxy are the next monitor candidates. Rows without an executable hedge stay as funding alerts, not paper-trade candidates.
