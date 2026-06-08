# Volume Price Dislocation Execution Gate

This joins supported volume-price dislocation labels to current Hyperliquid funding, spread, and public book depth. It is a rough paper gate, not a fill model.

| symbol | side | dir 1h | dir 4h | funding ann | spread bps | depth 10bps USD | 250 usage | net 4h bps | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | long_reversal | 0.01957757 | 0.04969039 | 0.109500 | 0.15764541 | 112794.91236000 | 0.00221641 | 488.24625459 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| INJ | long_reversal | 0.00300905 | 0.04554181 | -0.157592 | 5.85994726 | 11708.91084000 | 0.02135126 | 442.27774874 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| EIGEN | long_reversal | -0.00720621 | 0.03935698 | 0.025973 | 5.55092978 | 5443.43401800 | 0.04592689 | 379.90027022 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| PUMP | long_reversal | 0.00197239 | 0.02038133 | 0.109500 | 6.16332820 | 11909.81360250 | 0.02099109 | 189.14997180 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| FARTCOIN | long_reversal | 0.01072527 | 0.01116484 | 0.109500 | 3.41968026 | 18209.66735700 | 0.01372897 | 99.72871974 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| XPL | long_reversal | -0.00215268 | 0.01031553 | -0.097873 | 1.71609988 | 22117.24417000 | 0.01130340 | 93.88610812 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SOL | long_reversal | 0.00104408 | 0.00641579 | -0.125588 | 0.14990818 | 241280.36042500 | 0.00103614 | 56.58145182 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| LINK | long_reversal | 0.00114110 | 0.00588302 | 0.109500 | 2.00330545 | 59024.04936000 | 0.00423556 | 48.32689455 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ETH | long_reversal | 0.00363203 | 0.00458470 | -0.140761 | 0.59187358 | 11609416.23498500 | 0.00002153 | 37.89787042 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| VIRTUAL | long_reversal | 0.00199352 | 0.00381368 | -0.741977 | 3.27295591 | 15923.00398650 | 0.01570056 | 30.25186809 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| RENDER | long_reversal | 0.00552918 | 0.00318529 | -0.115541 | 2.44498778 | 2520.09440000 | 0.09920263 | 21.93549622 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TRUMP | long_reversal | 0.00078593 | 0.00253915 | -0.209832 | 2.38095238 | 40239.69600000 | 0.00621277 | 15.96868362 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| DOGE | long_reversal | 0.00094369 | 0.00236506 | 0.031385 | 0.11591045 | 120458.85673400 | 0.00207540 | 15.39137755 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ARB | long_reversal | 0.00121095 | 0.00254299 | -0.066340 | 2.44230065 | 23750.81055900 | 0.01052596 | 15.29052335 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TIA | long_reversal | -0.00258235 | 0.00393651 | 0.109500 | 4.74270809 | 8246.61760500 | 0.03031546 | 26.12239191 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SAND | long_reversal | 0.00304780 | 0.00259347 | 0.109500 | 11.36615391 | 1116.18847550 | 0.22397651 | 6.06854609 | thin_volume_watch | 24h notional volume is low for repeat observation |
| TAO | long_reversal | 0.00131585 | 0.00070492 | -0.153847 | 2.32585184 | 40649.83772500 | 0.00615009 | -2.57415184 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| FIL | long_reversal | -0.00149054 | 0.00054437 | -0.065248 | 3.88555738 | 6605.30715900 | 0.03784835 | -6.14392138 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| UNI | long_reversal | 0.00031287 | 0.00015643 | 0.109500 | 0.39849369 | 21018.40036500 | 0.01189434 | -7.33419369 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| WIF | long_reversal | -0.00429863 | 0.00012643 | 0.109500 | 1.26510216 | 12979.82136000 | 0.01926067 | -8.50080216 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| DOT | long_reversal | -0.00065930 | 0.00006181 | 0.060246 | 5.13964413 | 17390.11451400 | 0.01437598 | -12.79664013 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| SUI | long_reversal | 0.00064799 | -0.00126952 | 0.109500 | 0.13257413 | 51668.00062800 | 0.00483858 | -21.32777413 | label_contradicted | 4h label does not support the market-breadth direction |
| ETHFI | long_reversal | -0.00648875 | -0.00229602 | 0.109500 | 4.57501389 | 2972.24452900 | 0.08411152 | -36.03521389 | label_contradicted | 4h label does not support the market-breadth direction |
| LTC | long_reversal | -0.00163513 | -0.00429806 | 0.109500 | 1.15937996 | 46305.35431500 | 0.00539894 | -52.63997996 | label_contradicted | 4h label does not support the market-breadth direction |
| OP | long_reversal | -0.00330169 | -0.00484936 | -0.142081 | 3.13201441 | 4143.19933200 | 0.06033984 | -58.97684241 | label_contradicted | 4h label does not support the market-breadth direction |
| ADA | long_reversal | -0.00659008 | -0.00646917 | -0.389898 | 1.17515718 | 116850.41172000 | 0.00213949 | -72.08650118 | label_contradicted | 4h label does not support the market-breadth direction |
| LDO | long_reversal | 0.00000000 | -0.00652342 | 0.109500 | 4.51841253 | 10284.61205800 | 0.02430816 | -78.25261253 | label_contradicted | 4h label does not support the market-breadth direction |
| SEI | long_reversal | -0.00435686 | -0.00764467 | -0.047736 | 2.64671451 | 5880.88739250 | 0.04251059 | -86.87544251 | label_contradicted | 4h label does not support the market-breadth direction |
| PENGU | long_reversal | 0.01005245 | -0.00990676 | 0.109500 | 2.95377345 | 34395.96904500 | 0.00726829 | -110.52137345 | label_contradicted | 4h label does not support the market-breadth direction |
| ZK | long_reversal | 0.00077042 | -0.01126733 | -0.084750 | 9.61538462 | 1242.74800000 | 0.20116709 | -129.90169662 | label_contradicted | 4h label does not support the market-breadth direction |
| ZEC | long_reversal | -0.00511957 | -0.01362943 | -2.197118 | 0.21953174 | 63703.77275000 | 0.00392441 | -134.48132774 | label_contradicted | 4h label does not support the market-breadth direction |
| DYDX | long_reversal | -0.00427085 | -0.01460357 | 0.109500 | 3.65163411 | 227.59673500 | 1.09843404 | -158.18733411 | label_contradicted | 4h label does not support the market-breadth direction |
| AAVE | long_reversal | -0.01030637 | -0.01740661 | -0.018822 | 2.21046815 | 40209.49145000 | 0.00621744 | -184.19062415 | label_contradicted | 4h label does not support the market-breadth direction |
| FET | long_reversal | -0.00216491 | -0.01736634 | 0.109500 | 5.72136931 | 7911.81228000 | 0.03159832 | -187.88476931 | label_contradicted | 4h label does not support the market-breadth direction |
| APT | long_reversal | -0.00691325 | -0.01805838 | -0.010013 | 2.98998356 | 11661.24139400 | 0.02143854 | -191.52806356 | label_contradicted | 4h label does not support the market-breadth direction |
| ALGO | long_reversal | -0.01130151 | -0.01903308 | 0.109500 | 7.50322149 | 200.93369250 | 1.24419154 | -206.33402149 | label_contradicted | 4h label does not support the market-breadth direction |
| JTO | long_momentum_watch | 0.00030844 | -0.01920455 | -0.134954 | 7.34987054 | 1596.00090000 | 0.15664152 | -206.77914254 | label_contradicted | 4h label does not support the market-breadth direction |
| WLD | wait_or_fade_watch | -0.01685918 | -0.02182776 | -0.131875 | 2.82206858 | 20756.45520900 | 0.01204445 | -229.70183658 | label_contradicted | 4h label does not support the market-breadth direction |

## Interpretation

`paper_execution_probe` only means the current public venue context does not obviously kill a small repeat observation. It excludes queue position, account fees, realized fills, stop behavior, and whether the 4h label repeats.
