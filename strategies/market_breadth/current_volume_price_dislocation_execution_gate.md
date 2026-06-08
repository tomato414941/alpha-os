# Volume Price Dislocation Execution Gate

This joins supported volume-price dislocation labels to current Hyperliquid funding, spread, and public book depth. It is a rough paper gate, not a fill model.

| symbol | side | dir 1h | dir 4h | funding ann | spread bps | depth 10bps USD | 250 usage | net 4h bps | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | long_reversal | 0.01957757 | 0.04969039 | 0.109500 | 1.77584050 | 119236.83480000 | 0.00209667 | 486.62805950 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| INJ | long_reversal | 0.00300905 | 0.04554181 | -0.039451 | 6.34417129 | 3434.77485000 | 0.07278497 | 441.25406871 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| PUMP | long_reversal | 0.00197239 | 0.02038133 | 0.087955 | 6.29524709 | 23123.35925100 | 0.01081158 | 189.11643291 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| FARTCOIN | long_reversal | 0.01072527 | 0.01116484 | 0.109500 | 4.30200043 | 19206.03015750 | 0.01301675 | 98.84639957 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| XPL | long_reversal | -0.00215268 | 0.01031553 | 0.109500 | 2.86172161 | 7110.61478400 | 0.03515870 | 91.79357839 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SOL | long_reversal | 0.00104408 | 0.00641579 | -0.047157 | 0.15045399 | 218602.37088000 | 0.00114363 | 56.22277401 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| LINK | long_reversal | 0.00114110 | 0.00588302 | 0.109500 | 2.26888850 | 67873.41036000 | 0.00368333 | 48.06131150 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ETH | long_reversal | 0.00363203 | 0.00458470 | 0.090592 | 0.59550395 | 8803463.75895000 | 0.00002840 | 36.83783605 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TIA | long_reversal | -0.00258235 | 0.00393651 | 0.109500 | 1.24595066 | 3304.49682400 | 0.07565448 | 29.61914934 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| VIRTUAL | long_reversal | 0.00199352 | 0.00381368 | -0.108567 | 3.24844630 | 12424.69052700 | 0.02012123 | 27.38409370 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| RENDER | long_reversal | 0.00552918 | 0.00318529 | 0.109500 | 3.59324470 | 1889.37870000 | 0.13231863 | 19.75965530 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TRUMP | long_reversal | 0.00078593 | 0.00253915 | -0.193904 | 1.78928220 | 44494.93770000 | 0.00561862 | 16.48762580 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| DOGE | long_reversal | 0.00094369 | 0.00236506 | 0.109500 | 1.27389273 | 189731.17982650 | 0.00131765 | 13.87670727 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ARB | long_reversal | 0.00121095 | 0.00254299 | 0.072066 | 3.60902256 | 9031.43981250 | 0.02768108 | 13.49180944 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SAND | long_reversal | 0.00304780 | 0.00259347 | 0.109500 | 3.04814159 | 3635.73662400 | 0.06876186 | 14.38655841 | thin_volume_watch | 24h notional volume is low for repeat observation |
| EIGEN | long_reversal | -0.00720621 | 0.03935698 | 0.109500 | 10.91703057 | 2367.54489600 | 0.10559462 | 374.15276943 | wide_spread_watch | current spread is wide for a small directional repeat |
| TAO | long_reversal | 0.00131585 | 0.00070492 | 0.044371 | 0.47097610 | 23625.82740000 | 0.01058164 | -1.62438410 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| FIL | long_reversal | -0.00149054 | 0.00054437 | 0.109500 | 7.20832046 | 1389.83832000 | 0.17987704 | -10.26462046 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| UNI | long_reversal | 0.00031287 | 0.00015643 | 0.109500 | 3.95069532 | 18168.19424000 | 0.01376031 | -10.88639532 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| WIF | long_reversal | -0.00429863 | 0.00012643 | -0.020882 | 4.33020940 | 12812.45199000 | 0.01951227 | -10.97055740 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| DOT | long_reversal | -0.00065930 | 0.00006181 | 0.045627 | 6.85635927 | 15123.26525900 | 0.01653082 | -14.44660327 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| SUI | long_reversal | 0.00064799 | -0.00126952 | 0.109500 | 3.16664468 | 83414.47400000 | 0.00299708 | -24.36184468 | label_contradicted | 4h label does not support the market-breadth direction |
| ETHFI | long_reversal | -0.00648875 | -0.00229602 | 0.109500 | 9.52615587 | 5278.02932250 | 0.04736616 | -40.98635587 | label_contradicted | 4h label does not support the market-breadth direction |
| LTC | long_reversal | -0.00163513 | -0.00429806 | 0.109500 | 2.78661496 | 52686.71924000 | 0.00474503 | -54.26721496 | label_contradicted | 4h label does not support the market-breadth direction |
| OP | long_reversal | -0.00330169 | -0.00484936 | 0.109500 | 7.09040263 | 1524.92609500 | 0.16394237 | -64.08400263 | label_contradicted | 4h label does not support the market-breadth direction |
| ADA | long_reversal | -0.00659008 | -0.00646917 | -0.069476 | 1.20213981 | 79981.04654000 | 0.00312574 | -73.57659981 | label_contradicted | 4h label does not support the market-breadth direction |
| LDO | long_reversal | 0.00000000 | -0.00652342 | 0.109500 | 3.30693906 | 11290.32295950 | 0.02214286 | -77.04113906 | label_contradicted | 4h label does not support the market-breadth direction |
| SEI | long_reversal | -0.00435686 | -0.00764467 | 0.034978 | 2.20284167 | 11179.80941750 | 0.02236174 | -86.80925767 | label_contradicted | 4h label does not support the market-breadth direction |
| PENGU | long_reversal | 0.01005245 | -0.00990676 | 0.109500 | 1.45211646 | 39166.32830550 | 0.00638303 | -109.01971646 | label_contradicted | 4h label does not support the market-breadth direction |
| ZK | long_reversal | 0.00077042 | -0.01126733 | -0.323024 | 8.51023592 | 1109.87757400 | 0.22525007 | -127.70853992 | label_contradicted | 4h label does not support the market-breadth direction |
| ZEC | long_reversal | -0.00511957 | -0.01362943 | -1.149952 | 0.23780930 | 22337.22560000 | 0.01119208 | -139.28118530 | label_contradicted | 4h label does not support the market-breadth direction |
| DYDX | long_reversal | -0.00427085 | -0.01460357 | 0.109500 | 7.06015250 | 1165.68303600 | 0.21446653 | -161.59585250 | label_contradicted | 4h label does not support the market-breadth direction |
| AAVE | long_reversal | -0.01030637 | -0.01740661 | 0.109500 | 2.65510913 | 53032.69770000 | 0.00471407 | -185.22120913 | label_contradicted | 4h label does not support the market-breadth direction |
| FET | long_reversal | -0.00216491 | -0.01736634 | 0.109500 | 3.76612372 | 6535.10130000 | 0.03825495 | -185.92952372 | label_contradicted | 4h label does not support the market-breadth direction |
| APT | long_reversal | -0.00691325 | -0.01805838 | 0.023378 | 2.95202952 | 11612.63455000 | 0.02152828 | -191.64257752 | label_contradicted | 4h label does not support the market-breadth direction |
| JTO | long_momentum_watch | 0.00030844 | -0.01920455 | -0.324503 | 7.04811943 | 4249.47396000 | 0.05883081 | -205.61187143 | label_contradicted | 4h label does not support the market-breadth direction |
| ALGO | long_reversal | -0.01130151 | -0.01903308 | -0.050544 | 8.73241536 | 2312.54918100 | 0.10810581 | -206.83241936 | label_contradicted | 4h label does not support the market-breadth direction |
| WLD | wait_or_fade_watch | -0.01685918 | -0.02182776 | -0.315168 | 5.20584790 | 10457.82378200 | 0.02390555 | -232.92257190 | label_contradicted | 4h label does not support the market-breadth direction |

## Interpretation

`paper_execution_probe` only means the current public venue context does not obviously kill a small repeat observation. It excludes queue position, account fees, realized fills, stop behavior, and whether the 4h label repeats.
