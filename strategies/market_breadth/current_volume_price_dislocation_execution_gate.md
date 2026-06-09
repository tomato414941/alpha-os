# Volume Price Dislocation Execution Gate

This joins supported volume-price dislocation labels to current Hyperliquid funding, spread, and public book depth. It is a rough paper gate, not a fill model.

| symbol | side | dir 1h | dir 4h | funding ann | spread bps | depth 10bps USD | 250 usage | net 4h bps | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | long_reversal | 0.01957757 | 0.04969039 | 0.109500 | 0.15648350 | 119077.72816500 | 0.00209947 | 488.24741650 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| INJ | long_reversal | 0.00300905 | 0.04554181 | 0.086189 | 3.37894915 | 2068.15779000 | 0.12088053 | 443.64559485 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| EIGEN | long_reversal | -0.00720621 | 0.03935698 | 0.109500 | 5.43330617 | 1180.49301900 | 0.21177592 | 379.63649383 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| PUMP | long_reversal | 0.00197239 | 0.02038133 | 0.109500 | 6.15574023 | 21970.83291300 | 0.01137872 | 189.15755977 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| FARTCOIN | long_reversal | 0.01072527 | 0.01116484 | 0.109500 | 1.71203561 | 7040.54280600 | 0.03550863 | 101.43636439 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| XPL | long_reversal | -0.00215268 | 0.01031553 | 0.101393 | 1.71853295 | 5661.15419800 | 0.04416061 | 92.97378705 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SOL | long_reversal | 0.00104408 | 0.00641579 | -0.157482 | 0.15003413 | 486762.56965000 | 0.00051360 | 56.72696187 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| LINK | long_reversal | 0.00114110 | 0.00588302 | 0.109500 | 1.50061901 | 28289.12592000 | 0.00883732 | 48.82958099 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ETH | long_reversal | 0.00363203 | 0.00458470 | -0.256697 | 0.59299671 | 6448827.02182000 | 0.00003877 | 38.42613529 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| VIRTUAL | long_reversal | 0.00199352 | 0.00381368 | -0.363406 | 1.20247022 | 15939.96235650 | 0.01568385 | 30.59371778 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| RENDER | long_reversal | 0.00552918 | 0.00318529 | -0.183832 | 1.83010523 | 4806.11707500 | 0.05201704 | 22.86221077 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TRUMP | long_reversal | 0.00078593 | 0.00253915 | -0.440838 | 0.59476016 | 39095.75901000 | 0.00639456 | 18.80969984 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| DOGE | long_reversal | 0.00094369 | 0.00236506 | 0.096720 | 0.11598708 | 110364.96570150 | 0.00226521 | 15.09296892 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TIA | long_reversal | -0.00258235 | 0.00393651 | 0.109500 | 2.84328747 | 9362.15569500 | 0.02670325 | 28.02181253 | thin_volume_watch | 24h notional volume is low for repeat observation |
| SAND | long_reversal | 0.00304780 | 0.00259347 | 0.109500 | 6.16131082 | 2223.26715900 | 0.11244712 | 11.27338918 | thin_volume_watch | 24h notional volume is low for repeat observation |
| ARB | long_reversal | 0.00121095 | 0.00254299 | -0.061256 | 2.44081035 | 755.05251800 | 0.33110280 | 15.26879765 | too_large_for_visible_depth | 250 USD uses too much visible 10 bps depth |
| TAO | long_reversal | 0.00131585 | 0.00070492 | -0.193397 | 3.24999420 | 36044.46436500 | 0.00693588 | -3.31770220 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| FIL | long_reversal | -0.00149054 | 0.00054437 | -0.046505 | 2.97263240 | 1451.04386500 | 0.17228976 | -5.31658040 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| UNI | long_reversal | 0.00031287 | 0.00015643 | 0.109500 | 0.39932115 | 13358.17035000 | 0.01871514 | -7.33502115 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| WIF | long_reversal | -0.00429863 | 0.00012643 | 0.109500 | 5.05529226 | 11844.06300000 | 0.02110762 | -12.29099226 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| DOT | long_reversal | -0.00065930 | 0.00006181 | 0.033048 | 7.19557575 | 18355.55688800 | 0.01361985 | -14.72837975 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| SUI | long_reversal | 0.00064799 | -0.00126952 | 0.109500 | 0.92794507 | 69032.23301800 | 0.00362150 | -22.12314507 | label_contradicted | 4h label does not support the market-breadth direction |
| ETHFI | long_reversal | -0.00648875 | -0.00229602 | 0.109500 | 3.92953042 | 4463.98376400 | 0.05600379 | -35.38973042 | label_contradicted | 4h label does not support the market-breadth direction |
| LTC | long_reversal | -0.00163513 | -0.00429806 | 0.109500 | 1.62482736 | 39142.98927000 | 0.00638684 | -53.10542736 | label_contradicted | 4h label does not support the market-breadth direction |
| OP | long_reversal | -0.00330169 | -0.00484936 | -0.399079 | 1.04926289 | 3725.31043150 | 0.06710850 | -55.72058289 | label_contradicted | 4h label does not support the market-breadth direction |
| LDO | long_reversal | 0.00000000 | -0.00652342 | 0.109500 | 2.64096131 | 9087.72874100 | 0.02750962 | -76.37516131 | label_contradicted | 4h label does not support the market-breadth direction |
| ADA | long_reversal | -0.00659008 | -0.00646917 | -0.364363 | 9.99441489 | 20168.84453000 | 0.01239536 | -81.02235889 | label_contradicted | 4h label does not support the market-breadth direction |
| SEI | long_reversal | -0.00435686 | -0.00764467 | 0.109500 | 2.24162701 | 8410.36438500 | 0.02972523 | -87.18832701 | label_contradicted | 4h label does not support the market-breadth direction |
| PENGU | long_reversal | 0.01005245 | -0.00990676 | 0.109500 | 4.41014333 | 35179.05392250 | 0.00710650 | -111.97774333 | label_contradicted | 4h label does not support the market-breadth direction |
| ZK | long_reversal | 0.00077042 | -0.01126733 | -0.016297 | 7.72872186 | 438.82029400 | 0.56970929 | -128.32760586 | label_contradicted | 4h label does not support the market-breadth direction |
| ZEC | long_reversal | -0.00511957 | -0.01362943 | -1.984868 | 0.21895493 | 35879.53040000 | 0.00696776 | -135.44993093 | label_contradicted | 4h label does not support the market-breadth direction |
| DYDX | long_reversal | -0.00427085 | -0.01460357 | 0.109500 | 5.12538898 | 2179.99649250 | 0.11467908 | -159.66108898 | label_contradicted | 4h label does not support the market-breadth direction |
| FET | long_reversal | -0.00216491 | -0.01736634 | 0.109500 | 0.95606865 | 14392.06281000 | 0.01737069 | -183.11946865 | label_contradicted | 4h label does not support the market-breadth direction |
| AAVE | long_reversal | -0.01030637 | -0.01740661 | 0.109500 | 1.42196943 | 24723.31635000 | 0.01011191 | -183.98806943 | label_contradicted | 4h label does not support the market-breadth direction |
| APT | long_reversal | -0.00691325 | -0.01805838 | -0.098558 | 1.49153554 | 8460.97843250 | 0.02954741 | -189.62529954 | label_contradicted | 4h label does not support the market-breadth direction |
| ALGO | long_reversal | -0.01130151 | -0.01903308 | 0.109500 | 6.19972917 | 2629.19388150 | 0.09508618 | -205.03052917 | label_contradicted | 4h label does not support the market-breadth direction |
| JTO | long_momentum_watch | 0.00030844 | -0.01920455 | -0.009286 | 9.22300385 | 1202.80769500 | 0.20784702 | -209.22609985 | label_contradicted | 4h label does not support the market-breadth direction |
| WLD | wait_or_fade_watch | -0.01685918 | -0.02182776 | 0.096369 | 2.20407754 | 13867.99665500 | 0.01802712 | -228.04163754 | label_contradicted | 4h label does not support the market-breadth direction |

## Interpretation

`paper_execution_probe` only means the current public venue context does not obviously kill a small repeat observation. It excludes queue position, account fees, realized fills, stop behavior, and whether the 4h label repeats.
