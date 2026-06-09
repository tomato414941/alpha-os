# Volume Price Dislocation Execution Gate

This joins supported volume-price dislocation labels to current Hyperliquid funding, spread, and public book depth. It is a rough paper gate, not a fill model.

| symbol | side | dir 1h | dir 4h | funding ann | spread bps | depth 10bps USD | 250 usage | net 4h bps | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | long_reversal | 0.01957757 | 0.04969039 | 0.109500 | 0.15876418 | 118002.68829000 | 0.00211860 | 488.24513582 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| INJ | long_reversal | 0.00300905 | 0.04554181 | 0.072649 | 8.31751198 | 15472.12680000 | 0.01615809 | 438.76885602 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| EIGEN | long_reversal | -0.00720621 | 0.03935698 | 0.109500 | 5.51116010 | 1588.36975200 | 0.15739408 | 379.55863990 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| PUMP | long_reversal | 0.00197239 | 0.02038133 | 0.109500 | 6.27943485 | 12024.53752500 | 0.02079082 | 189.03386515 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| XPL | long_reversal | -0.00215268 | 0.01031553 | 0.043401 | 1.02441773 | 1403.52901000 | 0.17812243 | 93.93270627 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SOL | long_reversal | 0.00104408 | 0.00641579 | -0.181749 | 0.15220816 | 504269.28530500 | 0.00049577 | 56.83559584 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| LINK | long_reversal | 0.00114110 | 0.00588302 | 0.109500 | 3.17696320 | 64200.46027500 | 0.00389405 | 47.15323680 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ETH | long_reversal | 0.00363203 | 0.00458470 | -0.136307 | 0.59910733 | 10637400.64600500 | 0.00002350 | 37.87030067 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| VIRTUAL | long_reversal | 0.00199352 | 0.00381368 | -0.349717 | 2.27135731 | 19893.79644800 | 0.01256673 | 29.46232269 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| RENDER | long_reversal | 0.00552918 | 0.00318529 | -0.050283 | 3.09166795 | 3984.25710000 | 0.06274695 | 20.99083605 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TRUMP | long_reversal | 0.00078593 | 0.00253915 | -0.451498 | 2.41911098 | 15520.74310000 | 0.01610748 | 17.03402502 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ARB | long_reversal | 0.00121095 | 0.00254299 | -0.164793 | 1.23846678 | 2574.82078350 | 0.09709414 | 16.94391322 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| DOGE | long_reversal | 0.00094369 | 0.00236506 | 0.058323 | 0.46930180 | 97346.14349400 | 0.00256816 | 14.91498220 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TIA | long_reversal | -0.00258235 | 0.00393651 | 0.109500 | 6.43417836 | 1862.18027200 | 0.13425123 | 24.43092164 | thin_volume_watch | 24h notional volume is low for repeat observation |
| FARTCOIN | long_reversal | 0.01072527 | 0.01116484 | 0.109500 | 2.61631710 | 589.80236050 | 0.42387080 | 100.53208290 | too_large_for_visible_depth | 250 USD uses too much visible 10 bps depth |
| SAND | long_reversal | 0.00304780 | 0.00259347 | 0.109500 | 18.94031847 | 304.51547100 | 0.82097635 | -1.50561847 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| TAO | long_reversal | 0.00131585 | 0.00070492 | -0.071137 | 1.42190203 | 31819.28080500 | 0.00785687 | -2.04787403 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| WIF | long_reversal | -0.00429863 | 0.00012643 | 0.109500 | 1.28559491 | 9238.52445000 | 0.02706060 | -8.52129491 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| UNI | long_reversal | 0.00031287 | 0.00015643 | 0.109500 | 2.02367702 | 16491.02087500 | 0.01515976 | -8.95937702 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| FIL | long_reversal | -0.00149054 | 0.00054437 | 0.093313 | 12.17333255 | 225.21688200 | 1.11004112 | -15.15572055 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| DOT | long_reversal | -0.00065930 | 0.00006181 | 0.109500 | 9.90166087 | 3216.12206350 | 0.07773337 | -17.78356087 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| SUI | long_reversal | 0.00064799 | -0.00126952 | 0.109500 | 1.07934538 | 115629.19771200 | 0.00216208 | -22.27454538 | label_contradicted | 4h label does not support the market-breadth direction |
| ETHFI | long_reversal | -0.00648875 | -0.00229602 | 0.109500 | 19.34235977 | 197.78765600 | 1.26398181 | -50.80255977 | label_contradicted | 4h label does not support the market-breadth direction |
| LTC | long_reversal | -0.00163513 | -0.00429806 | 0.109500 | 3.29566855 | 25307.88480000 | 0.00987834 | -54.77626855 | label_contradicted | 4h label does not support the market-breadth direction |
| OP | long_reversal | -0.00330169 | -0.00484936 | -0.208858 | 3.18758965 | 2590.83536600 | 0.09649397 | -58.72750165 | label_contradicted | 4h label does not support the market-breadth direction |
| ADA | long_reversal | -0.00659008 | -0.00646917 | -0.349855 | 0.59575229 | 88209.81676000 | 0.00283415 | -71.68994029 | label_contradicted | 4h label does not support the market-breadth direction |
| LDO | long_reversal | 0.00000000 | -0.00652342 | 0.109500 | 6.87206506 | 4782.39651900 | 0.05227505 | -80.60626506 | label_contradicted | 4h label does not support the market-breadth direction |
| SEI | long_reversal | -0.00435686 | -0.00764467 | 0.109500 | 5.38168571 | 3377.58854400 | 0.07401730 | -90.32838571 | label_contradicted | 4h label does not support the market-breadth direction |
| PENGU | long_reversal | 0.01005245 | -0.00990676 | 0.109500 | 4.49068184 | 18150.20368650 | 0.01377395 | -112.05828184 | label_contradicted | 4h label does not support the market-breadth direction |
| ZK | long_reversal | 0.00077042 | -0.01126733 | 0.109500 | 7.83545544 | 445.60524000 | 0.56103470 | -129.00875544 | label_contradicted | 4h label does not support the market-breadth direction |
| ZEC | long_reversal | -0.00511957 | -0.01362943 | -2.022409 | 0.66509261 | 48421.82775000 | 0.00516296 | -135.72464861 | label_contradicted | 4h label does not support the market-breadth direction |
| DYDX | long_reversal | -0.00427085 | -0.01460357 | 0.109500 | 6.73577068 | 2272.25669000 | 0.11002278 | -161.27147068 | label_contradicted | 4h label does not support the market-breadth direction |
| FET | long_reversal | -0.00216491 | -0.01736634 | 0.109500 | 2.40946438 | 6660.60895500 | 0.03753411 | -184.57286438 | label_contradicted | 4h label does not support the market-breadth direction |
| AAVE | long_reversal | -0.01030637 | -0.01740661 | 0.109500 | 4.00393988 | 31656.31950000 | 0.00789732 | -186.57003988 | label_contradicted | 4h label does not support the market-breadth direction |
| APT | long_reversal | -0.00691325 | -0.01805838 | -0.186585 | 4.55338848 | 7157.83863900 | 0.03492674 | -192.28520048 | label_contradicted | 4h label does not support the market-breadth direction |
| JTO | long_momentum_watch | 0.00030844 | -0.01920455 | -0.140284 | 0.16969430 | 2690.72097000 | 0.09291190 | -199.57462630 | label_contradicted | 4h label does not support the market-breadth direction |
| ALGO | long_reversal | -0.01130151 | -0.01903308 | 0.109500 | 8.86995658 | 939.31237700 | 0.26615214 | -207.70075658 | label_contradicted | 4h label does not support the market-breadth direction |
| WLD | wait_or_fade_watch | -0.01685918 | -0.02182776 | 0.109500 | 4.64350969 | 42587.03510550 | 0.00587033 | -230.42110969 | label_contradicted | 4h label does not support the market-breadth direction |

## Interpretation

`paper_execution_probe` only means the current public venue context does not obviously kill a small repeat observation. It excludes queue position, account fees, realized fills, stop behavior, and whether the 4h label repeats.
