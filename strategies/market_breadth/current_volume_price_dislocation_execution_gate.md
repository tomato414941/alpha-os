# Volume Price Dislocation Execution Gate

This joins supported volume-price dislocation labels to current Hyperliquid funding, spread, and public book depth. It is a rough paper gate, not a fill model.

| symbol | side | dir 1h | dir 4h | funding ann | spread bps | depth 10bps USD | 250 usage | net 4h bps | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | long_reversal | 0.01957757 | 0.04969039 | 0.249711 | 1.69344100 | 428853.20604000 | 0.00058295 | 486.07022700 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| INJ | long_reversal | 0.00300905 | 0.04554181 | -0.538718 | 0.70353173 | 3916.24128000 | 0.06383672 | 449.17446827 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| FARTCOIN | long_reversal | 0.01072527 | 0.01116484 | 0.109500 | 6.67111408 | 8521.32332800 | 0.02933817 | 96.47728592 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| XPL | long_reversal | -0.00215268 | 0.01031553 | 0.109500 | 5.24181312 | 2179.67709800 | 0.11469589 | 89.41348688 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SOL | long_reversal | 0.00104408 | 0.00641579 | 0.109500 | 0.29565243 | 261887.24286000 | 0.00095461 | 55.36224757 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| LINK | long_reversal | 0.00114110 | 0.00588302 | 0.109500 | 4.31808228 | 4517.97783000 | 0.05533449 | 46.01211772 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ETH | long_reversal | 0.00363203 | 0.00458470 | 0.109500 | 1.76164890 | 2225355.28645500 | 0.00011234 | 35.58535110 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| VIRTUAL | long_reversal | 0.00199352 | 0.00381368 | -0.370275 | 4.20553280 | 7547.37901650 | 0.03312408 | 27.62201920 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TRUMP | long_reversal | 0.00078593 | 0.00253915 | -0.616546 | 2.34342961 | 34972.16203000 | 0.00714854 | 17.86335039 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ARB | long_reversal | 0.00121095 | 0.00254299 | 0.109500 | 3.55471296 | 7668.07906300 | 0.03260269 | 13.37518704 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| DOGE | long_reversal | 0.00094369 | 0.00236506 | 0.086536 | 2.52151887 | 168283.25373000 | 0.00148559 | 12.73394113 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SAND | long_reversal | 0.00304780 | 0.00259347 | -0.167200 | 3.20382952 | 4125.79693250 | 0.06059435 | 15.49434248 | thin_volume_watch | 24h notional volume is low for repeat observation |
| PUMP | long_reversal | 0.00197239 | 0.02038133 | 0.109500 | 12.32285890 | 162.30000000 | 1.54035736 | 182.99044110 | wide_spread_watch | current spread is wide for a small directional repeat |
| TIA | long_reversal | -0.00258235 | 0.00393651 | 0.109500 | 11.72079825 | 4174.13890800 | 0.05989259 | 19.14430175 | wide_spread_watch | current spread is wide for a small directional repeat |
| RENDER | long_reversal | 0.00552918 | 0.00318529 | -0.079882 | 14.27551749 | 1165.07160000 | 0.21457909 | 9.94214251 | wide_spread_watch | current spread is wide for a small directional repeat |
| TAO | long_reversal | 0.00131585 | 0.00070492 | 0.109500 | 4.15522057 | 33118.67507000 | 0.00754861 | -5.60602057 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| FIL | long_reversal | -0.00149054 | 0.00054437 | 0.109500 | 4.45800243 | 4999.94119250 | 0.05000059 | -7.51430243 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| WIF | long_reversal | -0.00429863 | 0.00012643 | 0.109500 | 3.65652995 | 5879.67288000 | 0.04251937 | -10.89222995 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| DOT | long_reversal | -0.00065930 | 0.00006181 | -0.018134 | 4.45750177 | 2232.32665000 | 0.11199078 | -11.75659777 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| UNI | long_reversal | 0.00031287 | 0.00015643 | 0.109500 | 5.04629001 | 9530.98215500 | 0.02623025 | -11.98199001 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| SUI | long_reversal | 0.00064799 | -0.00126952 | 0.109500 | 1.42593253 | 148353.35889000 | 0.00168517 | -22.62113253 | label_contradicted | 4h label does not support the market-breadth direction |
| ETHFI | long_reversal | -0.00648875 | -0.00229602 | 0.109500 | 3.20975766 | 4106.16669000 | 0.06088404 | -34.66995766 | label_contradicted | 4h label does not support the market-breadth direction |
| LTC | long_reversal | -0.00163513 | -0.00429806 | 0.109500 | 1.15123816 | 36691.79983000 | 0.00681351 | -52.63183816 | label_contradicted | 4h label does not support the market-breadth direction |
| AAVE | long_reversal | -0.00014164 | -0.00420208 | 0.073528 | 2.76940119 | 40138.27980000 | 0.00622847 | -53.12594519 | label_contradicted | 4h label does not support the market-breadth direction |
| OP | long_reversal | -0.00330169 | -0.00484936 | -0.045482 | 6.07287449 | 2316.25732000 | 0.10793274 | -62.35879449 | label_contradicted | 4h label does not support the market-breadth direction |
| APT | long_reversal | -0.00477042 | -0.00566488 | -0.365095 | 1.46832097 | 12256.99306000 | 0.02039652 | -64.45002097 | label_contradicted | 4h label does not support the market-breadth direction |
| ADA | long_reversal | -0.00659008 | -0.00646917 | -0.121370 | 0.58635551 | 53380.75554500 | 0.00468334 | -72.72385551 | label_contradicted | 4h label does not support the market-breadth direction |
| LDO | long_reversal | 0.00000000 | -0.00652342 | 0.109500 | 10.23055282 | 1447.43693400 | 0.17271910 | -83.96475282 | label_contradicted | 4h label does not support the market-breadth direction |
| SEI | long_reversal | -0.00435686 | -0.00764467 | -0.128104 | 3.76763601 | 10884.19898500 | 0.02296908 | -87.62938401 | label_contradicted | 4h label does not support the market-breadth direction |
| PENGU | long_reversal | 0.01005245 | -0.00990676 | 0.109500 | 4.30817836 | 28768.04293700 | 0.00869020 | -111.87577836 | label_contradicted | 4h label does not support the market-breadth direction |
| ALGO | long_reversal | -0.00565092 | -0.00987055 | 0.006505 | 10.54351287 | 1080.35909300 | 0.23140454 | -117.27871687 | label_contradicted | 4h label does not support the market-breadth direction |
| ZK | long_reversal | 0.00077042 | -0.01126733 | 0.109500 | 11.24121780 | 800.97727500 | 0.31211872 | -132.41451780 | label_contradicted | 4h label does not support the market-breadth direction |
| ZEC | long_reversal | -0.00511957 | -0.01362943 | -0.253490 | 3.60238658 | 38747.64600000 | 0.00645200 | -146.73919858 | label_contradicted | 4h label does not support the market-breadth direction |
| DYDX | long_reversal | -0.00427085 | -0.01460357 | 0.109500 | 17.98685576 | 492.97332000 | 0.50712684 | -172.52255576 | label_contradicted | 4h label does not support the market-breadth direction |
| FET | long_reversal | -0.00216491 | -0.01736634 | 0.109500 | 8.33487683 | 9786.22740000 | 0.02554611 | -190.49827683 | label_contradicted | 4h label does not support the market-breadth direction |
| JTO | long_momentum_watch | 0.00030844 | -0.01920455 | -0.141617 | 10.79820656 | 1546.36790000 | 0.16166916 | -210.19705456 | label_contradicted | 4h label does not support the market-breadth direction |
| WLD | wait_or_fade_watch | -0.01685918 | -0.02182776 | 0.109500 | 0.20510086 | 42628.29551500 | 0.00586465 | -225.98270086 | label_contradicted | 4h label does not support the market-breadth direction |
| EIGEN | long_reversal | -0.00720621 | 0.03935698 | 0.109500 | 26.83123155 | 0.00000000 |  | 358.23856845 | missing_l2_context | could not fetch current L2 context |

## Interpretation

`paper_execution_probe` only means the current public venue context does not obviously kill a small repeat observation. It excludes queue position, account fees, realized fills, stop behavior, and whether the 4h label repeats.
