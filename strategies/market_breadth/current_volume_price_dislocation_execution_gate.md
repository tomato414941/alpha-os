# Volume Price Dislocation Execution Gate

This joins supported volume-price dislocation labels to current Hyperliquid funding, spread, and public book depth. It is a rough paper gate, not a fill model.

| symbol | side | dir 1h | dir 4h | funding ann | spread bps | depth 10bps USD | 250 usage | net 4h bps | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | long_reversal | 0.01957757 | 0.04969039 | 0.109500 | 0.15768767 | 53111.31875000 | 0.00470709 | 488.24621233 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| INJ | long_reversal | 0.00300905 | 0.04554181 | -0.175246 | 4.90058807 | 2914.50736000 | 0.08577779 | 443.31772393 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| EIGEN | long_reversal | -0.00720621 | 0.03935698 | 0.109500 | 5.44810678 | 1388.01244200 | 0.18011366 | 379.62169322 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| PUMP | long_reversal | 0.00197239 | 0.02038133 | 0.109500 | 6.08087565 | 10437.27642100 | 0.02395261 | 189.23242435 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| FARTCOIN | long_reversal | 0.01072527 | 0.01116484 | 0.109500 | 0.84491572 | 2350.25808350 | 0.10637130 | 102.30348428 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| XPL | long_reversal | -0.00215268 | 0.01031553 | 0.109500 | 4.17280997 | 9161.95567800 | 0.02728675 | 90.48249003 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SOL | long_reversal | 0.00104408 | 0.00641579 | 0.024685 | 0.14838887 | 485119.27501500 | 0.00051534 | 55.89679513 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| LINK | long_reversal | 0.00114110 | 0.00588302 | 0.109500 | 1.36448494 | 66656.94686000 | 0.00375055 | 48.96571506 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ETH | long_reversal | 0.00363203 | 0.00458470 | -0.140041 | 0.59327816 | 13925818.07053500 | 0.00001795 | 37.89317784 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| VIRTUAL | long_reversal | 0.00199352 | 0.00381368 | -0.384719 | 0.33739330 | 19333.87536800 | 0.01293067 | 31.55611470 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TIA | long_reversal | -0.00258235 | 0.00393651 | 0.109500 | 4.65455448 | 7219.50943600 | 0.03462839 | 26.21054552 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| RENDER | long_reversal | 0.00552918 | 0.00318529 | 0.109500 | 4.79932809 | 2673.70760000 | 0.09350312 | 18.55357191 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TRUMP | long_reversal | 0.00078593 | 0.00253915 | -0.250497 | 0.58580592 | 39893.41709000 | 0.00626670 | 17.94951408 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| DOGE | long_reversal | 0.00094369 | 0.00236506 | 0.032243 | 1.49837772 | 179128.93843850 | 0.00139564 | 14.00499428 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SAND | long_reversal | 0.00304780 | 0.00259347 | -0.038292 | 13.90595384 | 613.14744000 | 0.40773227 | 4.20359416 | thin_volume_watch | 24h notional volume is low for repeat observation |
| ARB | long_reversal | 0.00121095 | 0.00254299 | -0.039519 | 1.20170642 | 859.13662450 | 0.29098981 | 16.40864558 | too_large_for_visible_depth | 250 USD uses too much visible 10 bps depth |
| TAO | long_reversal | 0.00131585 | 0.00070492 | -0.285606 | 2.29953779 | 15328.73263000 | 0.01630924 | -1.94620179 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| UNI | long_reversal | 0.00031287 | 0.00015643 | 0.109500 | 0.77217096 | 17346.67673000 | 0.01441198 | -7.70787096 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| WIF | long_reversal | -0.00429863 | 0.00012643 | 0.109500 | 4.34877147 | 11068.75822500 | 0.02258609 | -11.58447147 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| DOT | long_reversal | -0.00065930 | 0.00006181 | -0.039042 | 6.77085702 | 16830.50604750 | 0.01485398 | -13.97448502 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| FIL | long_reversal | -0.00149054 | 0.00054437 | -0.013368 | 17.29091972 | 3249.27539400 | 0.07694023 | -19.78617972 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| SUI | long_reversal | 0.00064799 | -0.00126952 | 0.109500 | 3.28845686 | 89471.44896200 | 0.00279419 | -24.48365686 | label_contradicted | 4h label does not support the market-breadth direction |
| ETHFI | long_reversal | -0.00648875 | -0.00229602 | 0.109500 | 5.21291500 | 3160.21266600 | 0.07910860 | -36.67311500 | label_contradicted | 4h label does not support the market-breadth direction |
| AAVE | long_reversal | -0.00014164 | -0.00420208 | 0.016513 | 1.71779716 | 38192.69326500 | 0.00654575 | -51.81400116 | label_contradicted | 4h label does not support the market-breadth direction |
| LTC | long_reversal | -0.00163513 | -0.00429806 | 0.109500 | 0.46086135 | 37525.81987000 | 0.00666208 | -51.94146135 | label_contradicted | 4h label does not support the market-breadth direction |
| APT | long_reversal | -0.00477042 | -0.00566488 | -0.037000 | 2.93815190 | 10796.84817300 | 0.02315491 | -67.41800390 | label_contradicted | 4h label does not support the market-breadth direction |
| OP | long_reversal | -0.00330169 | -0.00484936 | 0.109500 | 12.34060058 | 4767.60913200 | 0.05243718 | -69.33420058 | label_contradicted | 4h label does not support the market-breadth direction |
| ADA | long_reversal | -0.00659008 | -0.00646917 | -0.332522 | 1.75054705 | 89223.13762500 | 0.00280196 | -72.92388305 | label_contradicted | 4h label does not support the market-breadth direction |
| LDO | long_reversal | 0.00000000 | -0.00652342 | 0.109500 | 2.59437022 | 7669.86911600 | 0.03259508 | -76.32857022 | label_contradicted | 4h label does not support the market-breadth direction |
| SEI | long_reversal | -0.00435686 | -0.00764467 | -0.095301 | 4.22072375 | 12925.47278250 | 0.01934165 | -88.23225975 | label_contradicted | 4h label does not support the market-breadth direction |
| PENGU | long_reversal | 0.01005245 | -0.00990676 | 0.109500 | 2.89645185 | 27492.83061500 | 0.00909328 | -110.46405185 | label_contradicted | 4h label does not support the market-breadth direction |
| ALGO | long_reversal | -0.00565092 | -0.00987055 | -0.024971 | 12.26861817 | 1293.35348000 | 0.19329596 | -118.86009417 | label_contradicted | 4h label does not support the market-breadth direction |
| ZK | long_reversal | 0.00077042 | -0.01126733 | 0.109500 | 7.54432290 | 1526.40338400 | 0.16378370 | -128.71762290 | label_contradicted | 4h label does not support the market-breadth direction |
| ZEC | long_reversal | -0.00511957 | -0.01362943 | -1.016255 | 2.84261739 | 115191.02100000 | 0.00217031 | -142.49648139 | label_contradicted | 4h label does not support the market-breadth direction |
| DYDX | long_reversal | -0.00427085 | -0.01460357 | 0.109500 | 5.59518814 | 2477.19999000 | 0.10092039 | -160.13088814 | label_contradicted | 4h label does not support the market-breadth direction |
| FET | long_reversal | -0.00216491 | -0.01736634 | 0.109500 | 3.27615660 | 15494.55847000 | 0.01613470 | -185.43955660 | label_contradicted | 4h label does not support the market-breadth direction |
| JTO | long_momentum_watch | 0.00030844 | -0.01920455 | 0.109500 | 10.73961179 | 1807.70035000 | 0.13829726 | -211.28511179 | label_contradicted | 4h label does not support the market-breadth direction |
| WLD | wait_or_fade_watch | -0.01685918 | -0.02182776 | 0.109500 | 0.91283353 | 6703.24853550 | 0.03729535 | -226.69043353 | label_contradicted | 4h label does not support the market-breadth direction |

## Interpretation

`paper_execution_probe` only means the current public venue context does not obviously kill a small repeat observation. It excludes queue position, account fees, realized fills, stop behavior, and whether the 4h label repeats.
