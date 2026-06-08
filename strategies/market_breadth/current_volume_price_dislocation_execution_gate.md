# Volume Price Dislocation Execution Gate

This joins supported volume-price dislocation labels to current Hyperliquid funding, spread, and public book depth. It is a rough paper gate, not a fill model.

| symbol | side | dir 1h | dir 4h | funding ann | spread bps | depth 10bps USD | 250 usage | net 4h bps | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | long_reversal | 0.01957757 | 0.04969039 | 0.300543 | 1.23915737 | 111978.02880000 | 0.00223258 | 486.29239863 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| INJ | long_reversal | 0.00300905 | 0.04554181 | -0.499369 | 1.94396091 | 11085.09945000 | 0.02255280 | 447.75436309 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| PUMP | long_reversal | 0.00197239 | 0.02038133 | 0.109500 | 6.16332820 | 18018.98689250 | 0.01387425 | 189.14997180 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| FARTCOIN | long_reversal | 0.01072527 | 0.01116484 | 0.109500 | 5.04498444 | 11413.40288200 | 0.02190407 | 98.10341556 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| XPL | long_reversal | -0.00215268 | 0.01031553 | 0.109500 | 4.44401239 | 5486.28533700 | 0.04556817 | 90.21128761 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SOL | long_reversal | 0.00104408 | 0.00641579 | 0.109500 | 0.14834484 | 540612.66095500 | 0.00046244 | 55.50955516 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| LINK | long_reversal | 0.00114110 | 0.00588302 | 0.109500 | 2.23015165 | 59346.72648000 | 0.00421253 | 48.10004835 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ETH | long_reversal | 0.00363203 | 0.00458470 | 0.109500 | 0.59016200 | 10270698.90209000 | 0.00002434 | 36.75683800 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| VIRTUAL | long_reversal | 0.00199352 | 0.00381368 | -0.007207 | 2.87869680 | 13078.26862450 | 0.01911568 | 27.29101120 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TIA | long_reversal | -0.00258235 | 0.00393651 | 0.109500 | 5.57862766 | 3984.59287200 | 0.06274167 | 25.28647234 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| RENDER | long_reversal | 0.00552918 | 0.00318529 | -0.103701 | 5.94919388 | 3367.85124000 | 0.07423131 | 18.37722612 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| TRUMP | long_reversal | 0.00078593 | 0.00253915 | -0.442221 | 3.53648473 | 37019.64234000 | 0.00675317 | 15.87429127 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| ARB | long_reversal | 0.00121095 | 0.00254299 | 0.093905 | 1.18955570 | 8233.72961200 | 0.03036291 | 15.81155230 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| DOGE | long_reversal | 0.00094369 | 0.00236506 | 0.109500 | 0.80644697 | 151078.78746450 | 0.00165477 | 14.34415303 | paper_execution_probe | current public venue context does not obviously block a small repeat |
| SAND | long_reversal | 0.00304780 | 0.00259347 | -0.062674 | 3.21534286 | 4073.32610300 | 0.06137490 | 15.00554114 | thin_volume_watch | 24h notional volume is low for repeat observation |
| EIGEN | long_reversal | -0.00720621 | 0.03935698 | 0.109500 | 5.38068335 | 445.85786700 | 0.56071681 | 379.68911665 | too_large_for_visible_depth | 250 USD uses too much visible 10 bps depth |
| TAO | long_reversal | 0.00131585 | 0.00070492 | 0.109500 | 0.46278085 | 49150.47801500 | 0.00508642 | -1.91358085 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| FIL | long_reversal | -0.00149054 | 0.00054437 | 0.109500 | 3.96219301 | 1848.72114550 | 0.13522861 | -7.01849301 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| UNI | long_reversal | 0.00031287 | 0.00015643 | 0.109500 | 1.16838354 | 14698.51242500 | 0.01700852 | -8.10408354 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| WIF | long_reversal | -0.00429863 | 0.00012643 | 0.109500 | 1.22466475 | 15480.80814000 | 0.01614903 | -8.46036475 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| DOT | long_reversal | -0.00065930 | 0.00006181 | 0.036397 | 7.31432287 | 14727.55331800 | 0.01697499 | -14.86241887 | no_edge_after_rough_cost | 4h label is erased by rough funding, spread, and taker-fee assumptions |
| SUI | long_reversal | 0.00064799 | -0.00126952 | 0.109500 | 1.43589074 | 64131.20255000 | 0.00389826 | -22.63109074 | label_contradicted | 4h label does not support the market-breadth direction |
| ETHFI | long_reversal | -0.00648875 | -0.00229602 | 0.109500 | 8.33734167 | 10713.94978500 | 0.02333406 | -39.79754167 | label_contradicted | 4h label does not support the market-breadth direction |
| AAVE | long_reversal | -0.00014164 | -0.00420208 | 0.068586 | 3.55951745 | 36746.83485000 | 0.00680331 | -53.89349745 | label_contradicted | 4h label does not support the market-breadth direction |
| LTC | long_reversal | -0.00163513 | -0.00429806 | 0.109500 | 4.38733216 | 42393.59898000 | 0.00589712 | -55.86793216 | label_contradicted | 4h label does not support the market-breadth direction |
| OP | long_reversal | -0.00330169 | -0.00484936 | 0.109500 | 7.10912507 | 2393.95985200 | 0.10442949 | -64.10272507 | label_contradicted | 4h label does not support the market-breadth direction |
| APT | long_reversal | -0.00477042 | -0.00566488 | -0.080209 | 2.94681008 | 10567.77979400 | 0.02365681 | -67.22935808 | label_contradicted | 4h label does not support the market-breadth direction |
| ADA | long_reversal | -0.00659008 | -0.00646917 | -0.131325 | 1.18623962 | 94715.43360000 | 0.00263949 | -73.27828362 | label_contradicted | 4h label does not support the market-breadth direction |
| LDO | long_reversal | 0.00000000 | -0.00652342 | 0.109500 | 4.76426072 | 5241.10906050 | 0.04769983 | -78.49846072 | label_contradicted | 4h label does not support the market-breadth direction |
| SEI | long_reversal | -0.00435686 | -0.00764467 | 0.102702 | 2.58678155 | 6650.96363650 | 0.03758854 | -87.50244155 | label_contradicted | 4h label does not support the market-breadth direction |
| PENGU | long_reversal | 0.01005245 | -0.00990676 | 0.109500 | 2.89059113 | 21066.83973900 | 0.01186699 | -110.45819113 | label_contradicted | 4h label does not support the market-breadth direction |
| ALGO | long_reversal | -0.00565092 | -0.00987055 | -0.009958 | 7.74828543 | 1959.76260000 | 0.12756647 | -114.40831343 | label_contradicted | 4h label does not support the market-breadth direction |
| ZK | long_reversal | 0.00077042 | -0.01126733 | 0.109500 | 7.55073148 | 683.20798000 | 0.36592078 | -128.72403148 | label_contradicted | 4h label does not support the market-breadth direction |
| ZEC | long_reversal | -0.00511957 | -0.01362943 | 0.064578 | 2.03052557 | 73869.54510000 | 0.00338434 | -146.61970157 | label_contradicted | 4h label does not support the market-breadth direction |
| DYDX | long_reversal | -0.00427085 | -0.01460357 | 0.109500 | 6.28206471 | 768.54509250 | 0.32528996 | -160.81776471 | label_contradicted | 4h label does not support the market-breadth direction |
| FET | long_reversal | -0.00216491 | -0.01736634 | 0.109500 | 2.80308339 | 4435.75815000 | 0.05636015 | -184.96648339 | label_contradicted | 4h label does not support the market-breadth direction |
| JTO | long_momentum_watch | 0.00030844 | -0.01920455 | -0.012317 | 6.92792647 | 4090.83441000 | 0.06111223 | -206.91718247 | label_contradicted | 4h label does not support the market-breadth direction |
| WLD | wait_or_fade_watch | -0.01685918 | -0.02182776 | 0.109500 | 2.69947568 | 17729.71335750 | 0.01410062 | -228.47707568 | label_contradicted | 4h label does not support the market-breadth direction |

## Interpretation

`paper_execution_probe` only means the current public venue context does not obviously kill a small repeat observation. It excludes queue position, account fees, realized fills, stop behavior, and whether the 4h label repeats.
