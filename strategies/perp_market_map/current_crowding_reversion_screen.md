# Current Crowding Reversion Screen

This screen looks for perp states where funding carry and mark/oracle reversion point in the same direction. It is not a trade instruction.

| asset | action | annualized funding | mark/oracle | premium | OI/volume | impact spread | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MOVE | long_carry_reversion_watch | -7.953939 | -0.009098 | -0.006842 | 0.723731 | 0.004248 | 45.547189 | long perp receives funding and mark is below oracle |
| STABLE | long_carry_reversion_watch | -2.763176 | -0.004244 | -0.002649 | 2.924580 | 0.002296 | 19.431273 | long perp receives funding and mark is below oracle |
| HEMI | short_carry_reversion_watch | 0.109500 | 0.002166 | 0.000903 | 22.253094 | 0.003601 | 10.410551 | short perp receives funding and mark is above oracle |
| APEX | short_carry_reversion_watch | 0.109500 | 0.001244 | 0.000000 | 14.888845 | 0.004745 | 10.188222 | short perp receives funding and mark is above oracle |
| ETC | long_carry_reversion_watch | -0.372039 | -0.001458 | -0.001135 | 7.861831 | 0.001896 | 9.790565 | long perp receives funding and mark is below oracle |
| TRX | long_carry_reversion_watch | -0.203019 | -0.000888 | -0.000784 | 8.324127 | 0.000322 | 9.673020 | long perp receives funding and mark is below oracle |
| PURR | short_carry_reversion_watch | 0.109500 | 0.003794 | 0.000000 | 9.330542 | 0.013987 | 8.955471 | short perp receives funding and mark is above oracle |
| IP | long_carry_reversion_watch | -0.709831 | -0.001747 | -0.001294 | 4.438416 | 0.001297 | 8.575926 | long perp receives funding and mark is below oracle |
| STBL | short_carry_reversion_watch | 0.109500 | 0.001526 | 0.000509 | 7.670618 | 0.003203 | 8.115327 | short perp receives funding and mark is above oracle |
| VINE | short_carry_reversion_watch | 0.109500 | 0.000832 | 0.000000 | 7.844092 | 0.005175 | 7.924793 | short perp receives funding and mark is above oracle |
| ZEC | long_carry_reversion_watch | -0.741535 | -0.001402 | -0.001706 | 0.462026 | 0.000208 | 7.032751 | long perp receives funding and mark is below oracle |
| COMP | long_carry_reversion_watch | -0.447587 | -0.002738 | -0.001268 | 4.635035 | 0.003068 | 6.987039 | long perp receives funding and mark is below oracle |
| IO | long_carry_reversion_watch | -1.061861 | -0.002237 | -0.001010 | 0.816701 | 0.001591 | 6.971190 | long perp receives funding and mark is below oracle |
| W | long_carry_reversion_watch | -0.551368 | -0.003112 | -0.001037 | 3.414889 | 0.004167 | 6.281727 | long perp receives funding and mark is below oracle |
| ZORA | long_carry_reversion_watch | -0.257406 | -0.002054 | -0.000725 | 4.911598 | 0.003026 | 6.136301 | long perp receives funding and mark is below oracle |
| UMA | long_carry_reversion_watch | -0.216518 | -0.001812 | -0.001061 | 5.101478 | 0.001945 | 6.094220 | long perp receives funding and mark is below oracle |
| OP | long_carry_reversion_watch | -0.380057 | -0.001533 | -0.001329 | 3.746574 | 0.001639 | 6.019116 | long perp receives funding and mark is below oracle |
| FIL | long_carry_reversion_watch | -0.043599 | -0.001124 | -0.001085 | 5.363736 | 0.001152 | 5.638175 | long perp receives funding and mark is below oracle |
| XMR | short_carry_reversion_watch | 0.109500 | 0.000350 | 0.000044 | 4.884233 | 0.000543 | 5.623975 | short perp receives funding and mark is above oracle |
| TRUMP | long_carry_reversion_watch | -0.237702 | -0.001023 | -0.001095 | 3.949257 | 0.000717 | 5.471783 | long perp receives funding and mark is below oracle |
| kSHIB | long_carry_reversion_watch | -0.531050 | -0.001269 | -0.001057 | 2.243514 | 0.000847 | 5.452846 | long perp receives funding and mark is below oracle |
| SAND | long_carry_reversion_watch | -0.619573 | -0.001736 | -0.000733 | 1.969524 | 0.002822 | 5.181646 | long perp receives funding and mark is below oracle |
| kFLOKI | long_carry_reversion_watch | -0.010900 | -0.001014 | -0.000568 | 5.070471 | 0.001461 | 5.096457 | long perp receives funding and mark is below oracle |
| AIXBT | long_carry_reversion_watch | -0.757571 | -0.002185 | -0.001092 | 0.789735 | 0.002190 | 5.000079 | long perp receives funding and mark is below oracle |
| CC | short_carry_reversion_watch | 0.109500 | 0.000849 | 0.000000 | 4.218258 | 0.003189 | 4.657737 | short perp receives funding and mark is above oracle |

## Interpretation

`long_carry_reversion_watch` means long perp receives funding while the mark is below oracle. `short_carry_reversion_watch` means short perp receives funding while the mark is above oracle. OI/volume is a crowding proxy, not proof of forced liquidations or future returns.
