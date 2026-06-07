# Spot/Perp Carry Symbol Audit

This decomposes 14-day spot/perp carry candidates by symbol. Gross contribution excludes transaction costs; funding and basis contributions show whether the candidate is earning funding or relying on spot/perp basis movement.

## spot_perp_positive_funding_top_1_14d

| symbol | held steps | mean weight | gross contribution | funding contribution | basis contribution | mean funding | mean pair return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WIFUSDT | 112 | 1.000000 | 0.033175 | 0.032088 | 0.001087 | 0.000573 | 0.000296 |
| INJUSDT | 42 | 1.000000 | 0.018149 | 0.017183 | 0.000967 | 0.000818 | 0.000432 |
| APTUSDT | 84 | 1.000000 | 0.010762 | 0.012280 | -0.001518 | 0.000292 | 0.000128 |
| TRXUSDT | 56 | 1.000000 | 0.008866 | 0.007530 | 0.001336 | 0.000269 | 0.000158 |
| OPUSDT | 41 | 1.000000 | 0.007913 | 0.007408 | 0.000506 | 0.000361 | 0.000193 |
| FETUSDT | 70 | 1.000000 | 0.005136 | 0.006333 | -0.001197 | 0.000181 | 0.000073 |
| RUNEUSDT | 70 | 1.000000 | 0.004766 | 0.005604 | -0.000837 | 0.000160 | 0.000068 |
| AAVEUSDT | 42 | 1.000000 | 0.004258 | 0.004134 | 0.000123 | 0.000197 | 0.000101 |

## spot_perp_positive_funding_top_2_14d

| symbol | held steps | mean weight | gross contribution | funding contribution | basis contribution | mean funding | mean pair return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WIFUSDT | 168 | 0.500000 | 0.020180 | 0.019284 | 0.000896 | 0.000459 | 0.000240 |
| INJUSDT | 56 | 0.500000 | 0.012403 | 0.011966 | 0.000437 | 0.000855 | 0.000443 |
| FETUSDT | 112 | 0.562500 | 0.012244 | 0.013058 | -0.000814 | 0.000432 | 0.000207 |
| APTUSDT | 126 | 0.500000 | 0.009704 | 0.010888 | -0.001184 | 0.000346 | 0.000154 |
| AAVEUSDT | 84 | 0.500000 | 0.006873 | 0.006460 | 0.000414 | 0.000308 | 0.000164 |
| TRXUSDT | 70 | 0.500000 | 0.005079 | 0.004353 | 0.000727 | 0.000249 | 0.000145 |
| XRPUSDT | 28 | 0.500000 | 0.004847 | 0.004156 | 0.000691 | 0.000594 | 0.000346 |
| OPUSDT | 69 | 0.500000 | 0.004773 | 0.004233 | 0.000539 | 0.000245 | 0.000138 |

## spot_perp_positive_funding_top_3_14d

| symbol | held steps | mean weight | gross contribution | funding contribution | basis contribution | mean funding | mean pair return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WIFUSDT | 168 | 0.347222 | 0.013639 | 0.012931 | 0.000708 | 0.000459 | 0.000240 |
| FETUSDT | 168 | 0.388889 | 0.010539 | 0.011390 | -0.000851 | 0.000360 | 0.000173 |
| INJUSDT | 98 | 0.333333 | 0.010354 | 0.009359 | 0.000995 | 0.000573 | 0.000317 |
| APTUSDT | 168 | 0.333333 | 0.008838 | 0.009452 | -0.000614 | 0.000338 | 0.000158 |
| LINKUSDT | 84 | 0.361111 | 0.005217 | 0.004858 | 0.000359 | 0.000339 | 0.000179 |
| AAVEUSDT | 84 | 0.333333 | 0.004582 | 0.004306 | 0.000276 | 0.000308 | 0.000164 |
| ETCUSDT | 126 | 0.333333 | 0.004561 | 0.004564 | -0.000002 | 0.000217 | 0.000109 |
| RUNEUSDT | 140 | 0.483333 | 0.004227 | 0.004541 | -0.000314 | 0.000123 | 0.000055 |

## Interpretation

The useful follow-up is not just the best aggregate candidate; it is the symbols where funding contribution remains positive without depending on large favorable basis moves.
