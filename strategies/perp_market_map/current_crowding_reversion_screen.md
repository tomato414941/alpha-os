# Current Crowding Reversion Screen

This screen looks for perp states where funding carry and mark/oracle reversion point in the same direction. It is not a trade instruction.

| asset | action | annualized funding | mark/oracle | premium | OI/volume | impact spread | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PROVE | long_carry_reversion_watch | -5.137085 | -0.005068 | -0.004177 | 0.815971 | 0.001930 | 30.037069 | long perp receives funding and mark is below oracle |
| MOVE | long_carry_reversion_watch | -3.797444 | -0.004162 | -0.002984 | 2.095574 | 0.002840 | 21.796064 | long perp receives funding and mark is below oracle |
| STABLE | long_carry_reversion_watch | -2.772290 | -0.004418 | -0.002524 | 2.890325 | 0.002566 | 19.451172 | long perp receives funding and mark is below oracle |
| HEMI | short_carry_reversion_watch | 0.109500 | 0.001625 | 0.000542 | 21.996878 | 0.003243 | 10.383659 | short perp receives funding and mark is above oracle |
| APEX | short_carry_reversion_watch | 0.109500 | 0.002163 | 0.000703 | 14.751790 | 0.004069 | 10.366075 | short perp receives funding and mark is above oracle |
| SNX | long_carry_reversion_watch | -1.180505 | -0.002722 | -0.001829 | 3.727033 | 0.002241 | 10.118871 | long perp receives funding and mark is below oracle |
| AERO | long_carry_reversion_watch | -0.418060 | -0.002259 | -0.001079 | 7.505506 | 0.001787 | 10.015197 | long perp receives funding and mark is below oracle |
| ZRO | short_carry_reversion_watch | 0.109500 | 0.000370 | 0.000146 | 8.847277 | 0.000594 | 9.535054 | short perp receives funding and mark is above oracle |
| TRX | long_carry_reversion_watch | -0.268498 | -0.000949 | -0.000817 | 7.748555 | 0.000564 | 9.498967 | long perp receives funding and mark is below oracle |
| CC | short_carry_reversion_watch | 0.768679 | 0.002223 | 0.001414 | 4.379843 | 0.001510 | 9.210170 | short perp receives funding and mark is above oracle |
| AZTEC | long_carry_reversion_watch | -0.208806 | -0.002407 | -0.001520 | 8.116523 | 0.002859 | 9.206644 | long perp receives funding and mark is below oracle |
| PURR | short_carry_reversion_watch | 0.109500 | 0.010143 | 0.000000 | 8.994846 | 0.016286 | 9.028097 | short perp receives funding and mark is above oracle |
| ZK | long_carry_reversion_watch | -1.235399 | -0.002799 | -0.000772 | 2.187784 | 0.003967 | 8.967744 | long perp receives funding and mark is below oracle |
| PENDLE | long_carry_reversion_watch | -0.258339 | -0.001028 | -0.000220 | 7.005986 | 0.001747 | 8.488576 | long perp receives funding and mark is below oracle |
| ETC | long_carry_reversion_watch | -0.124108 | -0.001204 | -0.000977 | 7.662602 | 0.001404 | 8.315479 | long perp receives funding and mark is below oracle |
| UMA | long_carry_reversion_watch | -0.806243 | -0.002167 | -0.001436 | 3.973381 | 0.002041 | 7.750527 | long perp receives funding and mark is below oracle |
| VINE | short_carry_reversion_watch | 0.109500 | 0.003080 | 0.000000 | 7.598874 | 0.006981 | 7.724966 | short perp receives funding and mark is above oracle |
| TRUMP | long_carry_reversion_watch | -0.420300 | -0.001089 | -0.000841 | 4.873992 | 0.000721 | 7.486076 | long perp receives funding and mark is below oracle |
| STBL | short_carry_reversion_watch | 0.109500 | 0.001527 | 0.000000 | 6.588864 | 0.004453 | 6.902004 | short perp receives funding and mark is above oracle |
| ZEC | long_carry_reversion_watch | -0.725048 | -0.001697 | -0.001707 | 0.443338 | 0.000440 | 6.898385 | long perp receives funding and mark is below oracle |
| KAITO | long_carry_reversion_watch | -0.243677 | -0.001389 | -0.000547 | 5.007196 | 0.002052 | 6.228966 | long perp receives funding and mark is below oracle |
| ETHFI | short_carry_reversion_watch | 0.109500 | 0.000233 | 0.000000 | 5.578126 | 0.002064 | 6.016963 | short perp receives funding and mark is above oracle |
| FIL | long_carry_reversion_watch | -0.141626 | -0.000652 | -0.000430 | 5.163389 | 0.001370 | 5.916855 | long perp receives funding and mark is below oracle |
| ZORA | long_carry_reversion_watch | -0.207255 | -0.002547 | 0.000000 | 4.850145 | 0.005472 | 5.608343 | long perp receives funding and mark is below oracle |
| GRIFFAIN | short_carry_reversion_watch | 0.109500 | 0.001218 | 0.000000 | 4.923806 | 0.003771 | 5.221275 | short perp receives funding and mark is above oracle |

## Interpretation

`long_carry_reversion_watch` means long perp receives funding while the mark is below oracle. `short_carry_reversion_watch` means short perp receives funding while the mark is above oracle. OI/volume is a crowding proxy, not proof of forced liquidations or future returns.
