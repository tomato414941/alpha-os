# Volume Price Dislocation Execution Gate

This joins supported volume-price dislocation labels to current Hyperliquid funding, spread, and public book depth. It is a rough paper gate, not a fill model.

| symbol | side | dir 1h | dir 4h | funding ann | spread bps | depth 10bps USD | 250 usage | net 4h bps | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | long_reversal | 0.01957757 | 0.04969039 | 0.109500 | 3.82940022 | 84733.26927000 | 0.00295043 | 484.57449978 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| INJ | long_reversal | 0.00300905 | 0.04554181 | -0.399243 | 2.93906942 | 6721.58333000 | 0.03719362 | 446.30205858 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| FARTCOIN | long_reversal | 0.01072527 | 0.01116484 | 0.109500 | 5.21784503 | 9454.73077800 | 0.02644179 | 97.93055497 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| XPL | long_reversal | -0.00215268 | 0.01031553 | 0.109500 | 4.04285426 | 6227.54084400 | 0.04014426 | 90.61244574 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SOL | long_reversal | 0.00104408 | 0.00641579 | -0.169382 | 0.15208779 | 455000.38000000 | 0.00054945 | 56.77924421 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| LINK | long_reversal | 0.00114110 | 0.00588302 | 0.109500 | 2.55264837 | 57135.17050000 | 0.00437559 | 47.77755163 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ETH | long_reversal | 0.00363203 | 0.00458470 | -0.017248 | 0.59968217 | 9757794.89178000 | 0.00002562 | 37.32607383 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| VIRTUAL | long_reversal | 0.00199352 | 0.00381368 | -0.244802 | 4.68404389 | 7652.50301500 | 0.03266905 | 26.57057211 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TRUMP | long_reversal | 0.00078593 | 0.00253915 | -0.335081 | 0.60611571 | 8703.94866000 | 0.02872260 | 18.31543229 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| RENDER | long_reversal | 0.00552918 | 0.00318529 | 0.091771 | 5.45107659 | 3039.41794500 | 0.08225259 | 17.98277941 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| DOGE | long_reversal | 0.00094369 | 0.00236506 | 0.102510 | 1.17255288 | 136531.07031600 | 0.00183109 | 14.00996312 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ARB | long_reversal | 0.00121095 | 0.00254299 | 0.095739 | 3.65786746 | 4836.17850500 | 0.05169371 | 13.33486854 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TIA | long_reversal | -0.00258235 | 0.00393651 | 0.109500 | 7.25929900 | 4022.79062800 | 0.06214591 | 23.60580100 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SAND | long_reversal | 0.00304780 | 0.00259347 | -0.408839 | 3.27853045 | 3643.98629000 | 0.06860619 | 16.52301355 | thin_volume_watch | 24h notional volume is low for repeat observation |
| EIGEN | long_reversal | -0.00720621 | 0.03935698 | 0.109500 | 10.88139282 | 424.43647400 | 0.58901630 | 374.18840718 | wide_spread_watch | current spread is wide for a small directional repeat |
| PUMP | long_reversal | 0.00197239 | 0.02038133 | 0.024413 | 12.86173633 | 12388.88715000 | 0.02017938 | 182.84008767 | wide_spread_watch | current spread is wide for a small directional repeat |
| TAO | long_reversal | 0.00131585 | 0.00070492 | 0.109500 | 2.85932139 | 20396.86768000 | 0.01225678 | -4.31012139 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| FIL | long_reversal | -0.00149054 | 0.00054437 | -0.174163 | 5.47995251 | 1289.90169000 | 0.19381322 | -7.24098851 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| UNI | long_reversal | 0.00031287 | 0.00015643 | 0.109500 | 1.17488104 | 21242.91658500 | 0.01176863 | -8.11058104 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| WIF | long_reversal | -0.00429863 | 0.00012643 | -0.148846 | 3.80976570 | 16301.94739000 | 0.01533559 | -9.86580570 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| DOT | long_reversal | -0.00065930 | 0.00006181 | 0.002649 | 3.83918983 | 8489.72607950 | 0.02944736 | -11.23318583 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| SUI | long_reversal | 0.00064799 | -0.00126952 | 0.109500 | 1.60918307 | 100113.28286000 | 0.00249717 | -22.80438307 | label_contradicted | 4h label does not support the market-breadth direction |
| ETHFI | long_reversal | -0.00648875 | -0.00229602 | 0.109500 | 13.68468484 | 435.32606500 | 0.57428218 | -45.14488484 | label_contradicted | 4h label does not support the market-breadth direction |
| LTC | long_reversal | -0.00163513 | -0.00429806 | 0.005216 | 1.40567894 | 31363.77636000 | 0.00797098 | -52.41009494 | label_contradicted | 4h label does not support the market-breadth direction |
| OP | long_reversal | -0.00330169 | -0.00484936 | -0.153990 | 5.17732332 | 3296.60694000 | 0.07583555 | -60.96777132 | label_contradicted | 4h label does not support the market-breadth direction |
| ADA | long_reversal | -0.00659008 | -0.00646917 | -0.012622 | 3.08365969 | 53431.80399500 | 0.00467886 | -75.71772369 | label_contradicted | 4h label does not support the market-breadth direction |
| LDO | long_reversal | 0.00000000 | -0.00652342 | 0.109500 | 2.99087782 | 6966.40960800 | 0.03588649 | -76.72507782 | label_contradicted | 4h label does not support the market-breadth direction |
| SEI | long_reversal | -0.00435686 | -0.00764467 | -0.455092 | 2.23711372 | 6614.80902400 | 0.03779399 | -84.60576972 | label_contradicted | 4h label does not support the market-breadth direction |
| PENGU | long_reversal | 0.01005245 | -0.00990676 | 0.109500 | 2.94941749 | 17931.00508100 | 0.01394233 | -110.51701749 | label_contradicted | 4h label does not support the market-breadth direction |
| ZK | long_reversal | 0.00077042 | -0.01126733 | -1.379341 | 7.76849874 | 590.83745200 | 0.42312822 | -122.14343874 | label_contradicted | 4h label does not support the market-breadth direction |
| ZEC | long_reversal | -0.00511957 | -0.01362943 | -0.911005 | 1.16611276 | 131848.31250000 | 0.00189612 | -141.30057276 | label_contradicted | 4h label does not support the market-breadth direction |
| DYDX | long_reversal | -0.00427085 | -0.01460357 | 0.109500 | 8.99436123 | 375.16949950 | 0.66636547 | -163.53006123 | label_contradicted | 4h label does not support the market-breadth direction |
| AAVE | long_reversal | -0.01030637 | -0.01740661 | -0.079931 | 1.43162784 | 38234.79710000 | 0.00653855 | -183.13274384 | label_contradicted | 4h label does not support the market-breadth direction |
| FET | long_reversal | -0.00216491 | -0.01736634 | 0.109500 | 4.79570305 | 9980.18424000 | 0.02504964 | -186.95910305 | label_contradicted | 4h label does not support the market-breadth direction |
| APT | long_reversal | -0.00691325 | -0.01805838 | -0.213642 | 3.00210147 | 10944.87988400 | 0.02284173 | -190.61036547 | label_contradicted | 4h label does not support the market-breadth direction |
| ALGO | long_reversal | -0.01130151 | -0.01903308 | 0.069021 | 1.72076316 | 1703.24427600 | 0.14677871 | -200.36672716 | label_contradicted | 4h label does not support the market-breadth direction |
| JTO | long_momentum_watch | 0.00030844 | -0.01920455 | -0.049721 | 6.68907507 | 2498.40222000 | 0.10006395 | -206.50753907 | label_contradicted | 4h label does not support the market-breadth direction |
| WLD | wait_or_fade_watch | -0.01685918 | -0.02182776 | 0.099677 | 0.20880968 | 14461.01538000 | 0.01728786 | -226.03126168 | label_contradicted | 4h label does not support the market-breadth direction |

## Interpretation

`paper_execution_probe` only means the current public venue context does not obviously kill a small repeat observation. It excludes queue position, account fees, realized fills, stop behavior, and whether the 4h label repeats.
