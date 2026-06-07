# Current Funding Dislocation Monitor

This repeats the current dislocation watchlist over a short window. It is a persistence check, not a trade instruction.

| source | action | asset | long | short | obs | mean edge | min edge | mean net 8h | mean net 24h | positive net24 | liquidity | friction |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| okx_hl_current | paper_24h_monitor | STABLE | OkxSwap | HlPerp | 12 | 1.928168 | 1.916050 | -0.001308 | 0.002214 | 1.000000 | 11785.58 | 0.003069 |
| predicted_cross_venue | current_funding_monitor | STABLE | BybitPerp | HlPerp | 12 | 2.162954 | 2.141368 |  |  |  | 1178558.41 | 0.002539 |
| predicted_cross_venue | current_funding_monitor | SAGA | HlPerp | BybitPerp | 12 | 1.367043 | 1.359040 |  |  |  | 186610.78 | 0.002179 |
| hl_single_venue | current_funding_monitor | SAGA | HlPerp | cash_or_spot_proxy | 12 | 1.255674 | 1.242025 |  |  |  | 186638.28 | 0.002240 |
| predicted_cross_venue | current_funding_monitor | kNEIRO | HlPerp | BybitPerp | 12 | 1.228604 | 1.222440 |  |  |  | 292486.73 | 0.001230 |
| hl_single_venue | current_funding_monitor | kNEIRO | HlPerp | cash_or_spot_proxy | 12 | 1.118101 | 1.112236 |  |  |  | 291959.19 | 0.001230 |
| predicted_cross_venue | current_funding_monitor | ZEC | BybitPerp | HlPerp | 12 | 1.031012 | 0.953757 |  |  |  | 344015815.63 | 0.001094 |
| predicted_cross_venue | current_funding_monitor | WLD | BinPerp | HlPerp | 12 | 0.988624 | 0.983737 |  |  |  | 48755591.25 | 0.001073 |
| predicted_cross_venue | current_funding_monitor | AIXBT | HlPerp | BinPerp | 12 | 0.974453 | 0.968691 |  |  |  | 269715.38 | 0.003420 |
| predicted_cross_venue | current_funding_monitor | SNX | HlPerp | BinPerp | 12 | 0.909701 | 0.892505 |  |  |  | 249256.72 | 0.002109 |
| hl_single_venue | current_funding_monitor | SNX | HlPerp | cash_or_spot_proxy | 12 | 0.905871 | 0.890248 |  |  |  | 249092.80 | 0.002195 |
| hl_single_venue | current_funding_monitor | AIXBT | HlPerp | cash_or_spot_proxy | 12 | 0.865998 | 0.857489 |  |  |  | 269729.96 | 0.003348 |
| hl_single_venue | current_funding_monitor | BABY | HlPerp | cash_or_spot_proxy | 12 | 0.824540 | 0.799922 |  |  |  | 1737267.95 | 0.001643 |
| hl_single_venue | current_funding_monitor | MON | HlPerp | cash_or_spot_proxy | 12 | 0.730521 | 0.716918 |  |  |  | 2410831.97 | 0.001616 |
| hl_single_venue | current_funding_monitor | MORPHO | HlPerp | cash_or_spot_proxy | 12 | 0.720390 | 0.690385 |  |  |  | 801767.34 | 0.002896 |
| hl_single_venue | current_funding_monitor | BIO | HlPerp | cash_or_spot_proxy | 12 | 0.703430 | 0.695186 |  |  |  | 628515.95 | 0.001389 |
| predicted_cross_venue | current_funding_monitor | BSV | HlPerp | BybitPerp | 12 | 0.641360 | 0.627406 |  |  |  | 203495.92 | 0.003739 |
| predicted_cross_venue | current_funding_monitor | 0G | HlPerp | BybitPerp | 12 | 0.637075 | 0.616086 |  |  |  | 163985.29 | 0.001580 |
| predicted_cross_venue | current_funding_monitor | POPCAT | HlPerp | BybitPerp | 12 | 0.602321 | 0.590488 |  |  |  | 460696.51 | 0.001817 |
| hl_single_venue | current_funding_monitor | POPCAT | HlPerp | cash_or_spot_proxy | 12 | 0.593454 | 0.575785 |  |  |  | 460469.48 | 0.001869 |
| predicted_cross_venue | current_funding_monitor | PROVE | HlPerp | BinPerp | 12 | 0.591538 | 0.575119 |  |  |  | 135868.33 | 0.002281 |
| predicted_cross_venue | current_funding_monitor | MORPHO | HlPerp | BinPerp | 11 | 0.591177 | 0.562300 |  |  |  | 802490.91 | 0.002831 |
| okx_hl_current | paper_24h_monitor | WLD | OkxSwap | HlPerp | 10 | 0.832726 | 0.828313 | -0.000614 | 0.000907 | 1.000000 | 487826.06 | 0.001375 |
| hl_single_venue | current_funding_monitor | SEI | HlPerp | cash_or_spot_proxy | 9 | 0.584000 | 0.578983 |  |  |  | 914803.30 | 0.001222 |
| hl_single_venue | current_funding_monitor | STABLE | HlPerp | cash_or_spot_proxy | 9 | 0.578706 | 0.557024 |  |  |  | 1178212.13 | 0.002480 |

## Interpretation

Rows that appear in every sample with positive 24-hour proxy are the next monitor candidates. Rows without an executable hedge stay as funding alerts, not paper-trade candidates.
