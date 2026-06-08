# Volume Price Dislocation Execution Gate

This joins supported volume-price dislocation labels to current Hyperliquid funding, spread, and public book depth. It is a rough paper gate, not a fill model.

| symbol | side | dir 1h | dir 4h | funding ann | spread bps | depth 10bps USD | 250 usage | net 4h bps | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | long_reversal | 0.01957757 | 0.04969039 | 0.109500 | 0.65170992 | 139012.76730000 | 0.00179840 | 487.75219008 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| INJ | long_reversal | 0.00300905 | 0.04554181 | 0.103999 | 2.83080625 | 9510.22346000 | 0.02628750 | 444.11241375 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| EIGEN | long_reversal | -0.00720621 | 0.03935698 | 0.109500 | 5.45702592 | 4385.31360250 | 0.05700847 | 379.61277408 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| PUMP | long_reversal | 0.00197239 | 0.02038133 | 0.109500 | 6.32711167 | 15338.60709400 | 0.01629874 | 188.98618833 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| FARTCOIN | long_reversal | 0.01072527 | 0.01116484 | 0.109500 | 6.92580729 | 12451.52751100 | 0.02007786 | 96.22259271 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| XPL | long_reversal | -0.00215268 | 0.01031553 | 0.109500 | 5.31826969 | 2839.56077250 | 0.08804179 | 89.33703031 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SOL | long_reversal | 0.00104408 | 0.00641579 | -0.056371 | 0.15129852 | 315673.28050500 | 0.00079196 | 56.26400548 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| LINK | long_reversal | 0.00114110 | 0.00588302 | 0.109500 | 2.65468267 | 73246.15561500 | 0.00341315 | 47.67551733 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ETH | long_reversal | 0.00363203 | 0.00458470 | 0.074217 | 0.59792520 | 10665491.94568000 | 0.00002344 | 36.91018280 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TIA | long_reversal | -0.00258235 | 0.00393651 | 0.109500 | 0.93669503 | 9095.36161500 | 0.02748654 | 29.92840497 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| VIRTUAL | long_reversal | 0.00199352 | 0.00381368 | -0.063518 | 2.74122807 | 10111.84742400 | 0.02472347 | 27.68560793 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TRUMP | long_reversal | 0.00078593 | 0.00253915 | -0.029582 | 1.19631535 | 41417.17320000 | 0.00603614 | 16.33026065 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| RENDER | long_reversal | 0.00552918 | 0.00318529 | 0.109500 | 7.79727096 | 1157.07150000 | 0.21606271 | 15.55562904 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ARB | long_reversal | 0.00121095 | 0.00254299 | 0.027522 | 2.41925729 | 6858.47680700 | 0.03645124 | 14.88497071 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| DOGE | long_reversal | 0.00094369 | 0.00236506 | 0.109500 | 0.93008115 | 75609.48851800 | 0.00330646 | 14.22051885 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SAND | long_reversal | 0.00304780 | 0.00259347 | 0.109500 | 7.26091526 | 2504.85777000 | 0.09980607 | 10.17378474 | thin_volume_watch | 24h notional volume is low for repeat observation |
| TAO | long_reversal | 0.00131585 | 0.00070492 | 0.057857 | 2.36759240 | 31541.95804500 | 0.00792595 | -3.58258040 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| FIL | long_reversal | -0.00149054 | 0.00054437 | 0.109500 | 2.45806731 | 6947.87319900 | 0.03598223 | -5.51436731 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| WIF | long_reversal | -0.00429863 | 0.00012643 | -0.061865 | 1.86665837 | 13407.00601500 | 0.01864697 | -8.31987037 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| UNI | long_reversal | 0.00031287 | 0.00015643 | 0.109500 | 3.59274266 | 15973.70183000 | 0.01565072 | -10.52844266 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| DOT | long_reversal | -0.00065930 | 0.00006181 | 0.109500 | 5.95867964 | 18461.90879000 | 0.01354140 | -13.84057964 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| SUI | long_reversal | 0.00064799 | -0.00126952 | 0.109500 | 2.25567402 | 46813.65715250 | 0.00534032 | -23.45087402 | label_contradicted | 4h label does not support the market-breadth direction |
| ETHFI | long_reversal | -0.00648875 | -0.00229602 | 0.109500 | 2.30798398 | 3693.43552150 | 0.06768766 | -33.76818398 | label_contradicted | 4h label does not support the market-breadth direction |
| LTC | long_reversal | -0.00163513 | -0.00429806 | 0.109500 | 1.39759148 | 62714.46342000 | 0.00398632 | -52.87819148 | label_contradicted | 4h label does not support the market-breadth direction |
| AAVE | long_reversal | -0.00014164 | -0.00420208 | 0.109500 | 3.60928685 | 38905.08174000 | 0.00642590 | -54.13008685 | label_contradicted | 4h label does not support the market-breadth direction |
| OP | long_reversal | -0.00330169 | -0.00484936 | 0.109500 | 9.17945841 | 4405.32852650 | 0.05674946 | -66.17305841 | label_contradicted | 4h label does not support the market-breadth direction |
| APT | long_reversal | -0.00477042 | -0.00566488 | 0.109500 | 4.44675017 | 14363.03538650 | 0.01740579 | -69.59555017 | label_contradicted | 4h label does not support the market-breadth direction |
| ADA | long_reversal | -0.00659008 | -0.00646917 | -0.104905 | 1.81318183 | 93632.14268500 | 0.00267002 | -74.02586183 | label_contradicted | 4h label does not support the market-breadth direction |
| LDO | long_reversal | 0.00000000 | -0.00652342 | 0.109500 | 4.06722006 | 9986.98360300 | 0.02503258 | -77.80142006 | label_contradicted | 4h label does not support the market-breadth direction |
| SEI | long_reversal | -0.00435686 | -0.00764467 | 0.109500 | 2.81945423 | 5689.12231500 | 0.04394351 | -87.76615423 | label_contradicted | 4h label does not support the market-breadth direction |
| PENGU | long_reversal | 0.01005245 | -0.00990676 | 0.109500 | 4.37094777 | 37328.25663700 | 0.00669734 | -111.93854777 | label_contradicted | 4h label does not support the market-breadth direction |
| ZK | long_reversal | 0.00077042 | -0.01126733 | -0.260437 | 8.54985038 | 1234.23212500 | 0.20255509 | -128.03394238 | label_contradicted | 4h label does not support the market-breadth direction |
| ZEC | long_reversal | -0.00511957 | -0.01362943 | -1.192685 | 4.31292680 | 65261.01950000 | 0.00383077 | -143.16117480 | label_contradicted | 4h label does not support the market-breadth direction |
| DYDX | long_reversal | -0.00427085 | -0.01460357 | 0.109500 | 6.35840192 | 1008.98937800 | 0.24777268 | -160.89410192 | label_contradicted | 4h label does not support the market-breadth direction |
| FET | long_reversal | -0.00216491 | -0.01736634 | 0.109500 | 5.66598990 | 17225.51607000 | 0.01451335 | -187.82938990 | label_contradicted | 4h label does not support the market-breadth direction |
| ALGO | long_reversal | -0.01130151 | -0.01903308 | -0.011651 | 3.20598450 | 183.40700000 | 1.36308865 | -201.48358450 | label_contradicted | 4h label does not support the market-breadth direction |
| JTO | long_momentum_watch | 0.00030844 | -0.01920455 | -0.482796 | 5.57595648 | 5804.92336000 | 0.04306689 | -203.41690848 | label_contradicted | 4h label does not support the market-breadth direction |
| WLD | wait_or_fade_watch | -0.01685918 | -0.02182776 | -0.429178 | 5.82945603 | 15099.68848150 | 0.01655663 | -234.06677203 | label_contradicted | 4h label does not support the market-breadth direction |

## Interpretation

`paper_execution_probe` only means the current public venue context does not obviously kill a small repeat observation. It excludes queue position, account fees, realized fills, stop behavior, and whether the 4h label repeats.
