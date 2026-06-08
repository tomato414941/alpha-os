# Volume Price Dislocation Execution Gate

This joins supported volume-price dislocation labels to current Hyperliquid funding, spread, and public book depth. It is a rough paper gate, not a fill model.

| symbol | side | dir 1h | dir 4h | funding ann | spread bps | depth 10bps USD | 250 usage | net 4h bps | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | long_reversal | 0.01957757 | 0.04969039 | 0.109500 | 0.16285186 | 131309.52120000 | 0.00190390 | 488.24104814 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| INJ | long_reversal | 0.00300905 | 0.04554181 | -0.177166 | 4.62222551 | 4315.02097000 | 0.05793715 | 443.60485049 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| PUMP | long_reversal | 0.00197239 | 0.02038133 | -0.073093 | 6.56383328 | 12381.49973500 | 0.02019142 | 189.58322272 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| FARTCOIN | long_reversal | 0.01072527 | 0.01116484 | 0.109500 | 5.28262018 | 18790.73199000 | 0.01330443 | 97.86577982 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| XPL | long_reversal | -0.00215268 | 0.01031553 | 0.109500 | 6.13900460 | 6383.94048000 | 0.03916077 | 88.51629540 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SOL | long_reversal | 0.00104408 | 0.00641579 | -0.290631 | 0.15302804 | 517069.28197500 | 0.00048349 | 57.33195196 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| LINK | long_reversal | 0.00114110 | 0.00588302 | 0.084179 | 2.18650924 | 62841.03337500 | 0.00397829 | 48.25931076 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ETH | long_reversal | 0.00363203 | 0.00458470 | -0.111958 | 0.60413835 | 12734247.75885000 | 0.00001963 | 37.75408565 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| VIRTUAL | long_reversal | 0.00199352 | 0.00381368 | -0.544834 | 3.32083650 | 12836.87407800 | 0.01947515 | 29.30379150 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| RENDER | long_reversal | 0.00552918 | 0.00318529 | 0.109500 | 5.49970974 | 3299.08320000 | 0.07577863 | 17.85319026 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TRUMP | long_reversal | 0.00078593 | 0.00253915 | -0.621942 | 2.44304648 | 40536.27340000 | 0.00616732 | 17.78836952 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| DOGE | long_reversal | 0.00094369 | 0.00236506 | 0.012417 | 0.11781824 | 161091.52292800 | 0.00155191 | 15.47608176 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TIA | long_reversal | -0.00258235 | 0.00393651 | 0.080899 | 3.48062715 | 3984.72729750 | 0.06273955 | 27.51506885 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SAND | long_reversal | 0.00304780 | 0.00259347 | 0.109500 | 6.79459926 | 1139.64042600 | 0.21936744 | 10.64010074 | thin_volume_watch | 24h notional volume is low for repeat observation |
| EIGEN | long_reversal | -0.00720621 | 0.03935698 | 0.109500 | 10.92299290 | 1255.97628100 | 0.19904834 | 374.14680710 | wide_spread_watch | current spread is wide for a small directional repeat |
| TAO | long_reversal | 0.00131585 | 0.00070492 | 0.109500 | 2.38885836 | 28212.63956000 | 0.00886128 | -3.83965836 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| WIF | long_reversal | -0.00429863 | 0.00012643 | -0.214324 | 3.20338277 | 19152.56601000 | 0.01305308 | -8.96043477 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| FIL | long_reversal | -0.00149054 | 0.00054437 | 0.065085 | 6.58345184 | 5278.23410400 | 0.04736433 | -9.43694384 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| UNI | long_reversal | 0.00031287 | 0.00015643 | 0.109500 | 2.75454993 | 10038.44575000 | 0.02490425 | -9.69024993 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| DOT | long_reversal | -0.00065930 | 0.00006181 | -0.009717 | 5.74463529 | 16344.31868950 | 0.01529583 | -13.08216729 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| SUI | long_reversal | 0.00064799 | -0.00126952 | 0.094753 | 0.27084123 | 60910.00186800 | 0.00410442 | -21.39870523 | label_contradicted | 4h label does not support the market-breadth direction |
| ETHFI | long_reversal | -0.00648875 | -0.00229602 | 0.109500 | 10.08844201 | 623.85252300 | 0.40073574 | -41.54864201 | label_contradicted | 4h label does not support the market-breadth direction |
| LTC | long_reversal | -0.00163513 | -0.00429806 | 0.035438 | 1.41911069 | 40319.47640000 | 0.00620048 | -52.56152669 | label_contradicted | 4h label does not support the market-breadth direction |
| OP | long_reversal | -0.00330169 | -0.00484936 | -0.183221 | 6.28798994 | 2048.65785800 | 0.12203111 | -61.94496594 | label_contradicted | 4h label does not support the market-breadth direction |
| ADA | long_reversal | -0.00659008 | -0.00646917 | -0.176212 | 1.86225519 | 89488.75578500 | 0.00279365 | -73.74933519 | label_contradicted | 4h label does not support the market-breadth direction |
| LDO | long_reversal | 0.00000000 | -0.00652342 | 0.109500 | 2.63073830 | 13220.00748900 | 0.01891073 | -76.36493830 | label_contradicted | 4h label does not support the market-breadth direction |
| ARB | long_reversal | -0.00286978 | -0.00676987 | 0.073950 | 3.68346737 | 8823.97934600 | 0.02833189 | -79.71983937 | label_contradicted | 4h label does not support the market-breadth direction |
| SEI | long_reversal | -0.00435686 | -0.00764467 | -0.137695 | 2.66221599 | 10897.33520300 | 0.02294139 | -86.48017199 | label_contradicted | 4h label does not support the market-breadth direction |
| PENGU | long_reversal | 0.01005245 | -0.00990676 | 0.109500 | 4.46328944 | 28203.62908800 | 0.00886411 | -112.03088944 | label_contradicted | 4h label does not support the market-breadth direction |
| ZK | long_reversal | 0.00077042 | -0.01126733 | -0.922184 | 9.79719800 | 2409.59711100 | 0.10375178 | -126.25961400 | label_contradicted | 4h label does not support the market-breadth direction |
| ZEC | long_reversal | -0.00511957 | -0.01362943 | -1.011059 | 0.23618607 | 49054.54470000 | 0.00509637 | -139.91377807 | label_contradicted | 4h label does not support the market-breadth direction |
| DYDX | long_reversal | -0.00427085 | -0.01460357 | 0.109500 | 10.48108165 | 370.33868550 | 0.67505775 | -165.01678165 | label_contradicted | 4h label does not support the market-breadth direction |
| AAVE | long_reversal | -0.01030637 | -0.01740661 | -0.212845 | 2.72971780 | 33661.61152500 | 0.00742686 | -183.82392180 | label_contradicted | 4h label does not support the market-breadth direction |
| FET | long_reversal | -0.00216491 | -0.01736634 | 0.109500 | 3.39221245 | 8102.32272000 | 0.03085535 | -185.55561245 | label_contradicted | 4h label does not support the market-breadth direction |
| APT | long_reversal | -0.00691325 | -0.01805838 | -0.129762 | 7.59474444 | 9116.42920650 | 0.02742302 | -195.58602444 | label_contradicted | 4h label does not support the market-breadth direction |
| JTO | long_momentum_watch | 0.00030844 | -0.01920455 | 0.045859 | 6.55908644 | 2167.89337000 | 0.11531932 | -206.81398644 | label_contradicted | 4h label does not support the market-breadth direction |
| ALGO | long_reversal | -0.01130151 | -0.01903308 | -0.164546 | 9.54923281 | 2325.78265200 | 0.10749070 | -207.12868081 | label_contradicted | 4h label does not support the market-breadth direction |
| WLD | wait_or_fade_watch | -0.01685918 | -0.02182776 | 0.109500 | 0.41449059 | 14634.92810400 | 0.01708242 | -226.19209059 | label_contradicted | 4h label does not support the market-breadth direction |
| BAT | long_reversal | -0.00130368 | -0.01228385 | 0.000000 |  |  |  |  | not_hyperliquid | symbol is not in current Hyperliquid perp universe |

## Interpretation

`paper_execution_probe` only means the current public venue context does not obviously kill a small repeat observation. It excludes queue position, account fees, realized fills, stop behavior, and whether the 4h label repeats.
