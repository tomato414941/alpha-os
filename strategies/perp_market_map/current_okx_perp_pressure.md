# Current OKX Perp Pressure

This maps current OKX USDT swap funding, premium, open interest, volume, and near-touch spread. It is a candidate screen, not a deployable strategy.

| asset | action | ann funding | settled ann funding | premium | OI USD | volume USD | OI/vol | spread bps | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MOVE | long_carry_discount_watch | -6.095293 | -6.095157 | -0.009184 | 1840511 | 15504281 | 0.1187 | 6.8942 | 823.668519 |
| HOME | long_carry_discount_watch | -5.374167 | -5.333914 | -0.039633 | 3092906 | 66304392 | 0.0466 | 3.1701 | 818.421890 |
| BEAT | short_carry_watch | 2.248072 | 2.777536 | -0.001616 | 40172903 | 793485160 | 0.0506 | 0.2352 | 275.023083 |
| ZEC | long_carry_discount_watch | -1.095276 | -0.987629 | -0.001476 | 61157351 | 855384224 | 0.0715 | 0.2380 | 132.382293 |
| MU | short_carry_premium_watch | 1.005497 | 0.820633 | 0.001947 | 36416773 | 192681847 | 0.1890 | 0.1075 | 124.314869 |
| MRVL | short_carry_premium_watch | 0.667549 | 0.578493 | 0.001777 | 10939421 | 62684187 | 0.1745 | 3.4789 | 69.157008 |
| H | long_carry_discount_watch | -0.650620 | -0.574183 | -0.001301 | 16946836 | 73353199 | 0.2310 | 2.3539 | 61.026924 |
| DRAM | short_carry_premium_watch | 0.542524 | 0.225787 | 0.001932 | 5149463 | 20961125 | 0.2457 | 1.6663 | 52.390803 |
| CL | long_carry_discount_watch | -0.529259 | -0.541040 | -0.001248 | 24514967 | 165315239 | 0.1483 | 1.0955 | 52.179758 |
| BZ | long_carry_discount_watch | -0.474174 | -0.377344 | -0.002201 | 5750404 | 27314470 | 0.2105 | 1.0624 | 50.060190 |
| WLD | long_carry_discount_watch | -0.548983 | -0.190940 | -0.000694 | 32773475 | 411123858 | 0.0797 | 2.1661 | 47.843204 |
| SNDK | short_carry_premium_watch | 0.539718 | 0.391114 | 0.000501 | 11799931 | 69609364 | 0.1695 | 0.6125 | 37.432918 |
| QQQ | short_carry_premium_watch | 0.434955 | 0.510630 | 0.000658 | 9219045 | 15806231 | 0.5833 | 0.1394 | 28.981914 |
| RAVE | short_carry_premium_watch | 0.407682 | 0.428219 | 0.000436 | 3595341 | 14253099 | 0.2522 | 2.7820 | 23.258191 |
| TRX | long_carry_discount_watch | -0.320531 | -0.300470 | -0.000699 | 14920775 | 14673308 | 1.0169 | 0.3058 | 22.234976 |
| MEGA | long_carry_discount_watch | -0.282300 | -0.299859 | -0.001289 | 1189580 | 13730915 | 0.0866 | 1.9978 | 20.112254 |
| EWY | short_carry_premium_watch | 0.331342 | 0.244028 | 0.000355 | 1787847 | 15689168 | 0.1140 | 2.7045 | 17.524885 |
| BILL | short_carry_premium_watch | 0.250449 | 0.299789 | 0.000857 | 3356837 | 16151814 | 0.2078 | 1.3713 | 16.818593 |
| HYPE | long_carry_discount_watch | -0.144227 | -0.027105 | -0.001095 | 87428602 | 472484141 | 0.1850 | 1.6159 | 15.357347 |
| IP | long_carry_discount_watch | -0.218775 | -0.189535 | -0.000722 | 3786322 | 21461336 | 0.1764 | 3.1294 | 14.330483 |
| CBRS | short_carry_premium_watch | 0.123103 | 0.109500 | 0.002963 | 1399798 | 16453990 | 0.0851 | 4.6740 | 13.500447 |
| SOXL | short_carry_watch | 0.247628 | 0.289289 | 0.000000 | 3176535 | 30338192 | 0.1047 | 4.8674 | 11.997833 |
| SAHARA | short_carry_premium_watch | 0.154289 | 0.176571 | 0.001044 | 6466759 | 28749656 | 0.2249 | 2.6059 | 11.902219 |
| SOL | long_carry_discount_watch | -0.117404 | -0.205154 | -0.000752 | 181159837 | 712605929 | 0.2542 | 1.5041 | 11.793866 |
| AAVE | long_carry_discount_watch | -0.153251 | -0.072517 | -0.000845 | 9208216 | 20695256 | 0.4449 | 1.5621 | 11.090856 |

## Interpretation

High scores mean the instrument is liquid enough to inspect and has large current funding or premium pressure. This does not include future-return labels, funding decay labels, fees, maker/taker fill probability, or liquidation data.
