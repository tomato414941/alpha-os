# Current Crowding Reversion Screen

This screen looks for perp states where funding carry and mark/oracle reversion point in the same direction. It is not a trade instruction.

| asset | action | annualized funding | mark/oracle | premium | OI/volume | impact spread | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MON | long_carry_reversion_watch | -0.739466 | -0.001966 | -0.001341 | 11.538740 | 0.001298 | 14.821547 | long perp receives funding and mark is below oracle |
| AERO | long_carry_reversion_watch | -0.288468 | -0.001183 | -0.000549 | 12.183732 | 0.001646 | 11.585710 | long perp receives funding and mark is below oracle |
| XAI | long_carry_reversion_watch | -0.296142 | -0.001266 | 0.000000 | 13.649106 | 0.005070 | 10.939232 | long perp receives funding and mark is below oracle |
| ZRO | short_carry_reversion_watch | 0.109500 | 0.000545 | 0.000589 | 10.302630 | 0.000400 | 10.728549 | short perp receives funding and mark is above oracle |
| HEMI | short_carry_reversion_watch | 0.109500 | 0.001079 | 0.001079 | 15.345254 | 0.002870 | 10.397096 | short perp receives funding and mark is above oracle |
| STBL | short_carry_reversion_watch | 0.109500 | 0.000565 | 0.000000 | 9.740925 | 0.004681 | 9.915925 | short perp receives funding and mark is above oracle |
| PURR | short_carry_reversion_watch | 0.109500 | 0.002849 | 0.001274 | 8.719279 | 0.009957 | 8.685959 | short perp receives funding and mark is above oracle |
| MORPHO | long_carry_reversion_watch | -0.696141 | -0.002162 | -0.001128 | 4.355470 | 0.001276 | 8.583529 | long perp receives funding and mark is below oracle |
| SNX | long_carry_reversion_watch | -0.854413 | -0.002191 | -0.000620 | 3.199742 | 0.003355 | 7.710219 | long perp receives funding and mark is below oracle |
| IP | long_carry_reversion_watch | -0.164939 | -0.001543 | -0.000868 | 6.351635 | 0.001191 | 7.327372 | long perp receives funding and mark is below oracle |
| MEGA | long_carry_reversion_watch | -0.133796 | -0.001565 | 0.000000 | 6.330594 | 0.002251 | 7.067058 | long perp receives funding and mark is below oracle |
| UMA | long_carry_reversion_watch | -1.027970 | -0.002097 | 0.000000 | 2.201910 | 0.003938 | 7.008908 | long perp receives funding and mark is below oracle |
| ATOM | long_carry_reversion_watch | -0.291938 | -0.001898 | -0.001127 | 5.366411 | 0.002377 | 6.995675 | long perp receives funding and mark is below oracle |
| MET | long_carry_reversion_watch | -0.020541 | -0.001218 | -0.000325 | 6.904011 | 0.002237 | 6.916876 | long perp receives funding and mark is below oracle |
| SAGA | long_carry_reversion_watch | -1.168327 | -0.001459 | -0.000729 | 0.707898 | 0.002191 | 6.816733 | long perp receives funding and mark is below oracle |
| BABY | long_carry_reversion_watch | -0.976564 | -0.002571 | -0.001928 | 0.344200 | 0.001804 | 6.563514 | long perp receives funding and mark is below oracle |
| XMR | short_carry_reversion_watch | 0.109500 | 0.000683 | 0.000329 | 5.673850 | 0.000738 | 6.426430 | short perp receives funding and mark is above oracle |
| STABLE | long_carry_reversion_watch | -0.697293 | -0.003107 | -0.001746 | 1.969573 | 0.002493 | 6.308339 | long perp receives funding and mark is below oracle |
| SEI | long_carry_reversion_watch | -0.597955 | -0.001637 | -0.001112 | 2.533926 | 0.000911 | 6.200026 | long perp receives funding and mark is below oracle |
| BIO | long_carry_reversion_watch | -0.663671 | -0.001605 | -0.001284 | 2.246593 | 0.001500 | 6.137644 | long perp receives funding and mark is below oracle |
| OP | long_carry_reversion_watch | -0.376521 | -0.001666 | -0.000833 | 3.866508 | 0.002191 | 6.053407 | long perp receives funding and mark is below oracle |
| TRUMP | long_carry_reversion_watch | -0.180495 | -0.001043 | -0.000577 | 4.893756 | 0.000896 | 6.018081 | long perp receives funding and mark is below oracle |
| BSV | long_carry_reversion_watch | -0.633889 | -0.003821 | -0.001728 | 2.546054 | 0.003352 | 6.002426 | long perp receives funding and mark is below oracle |
| AIXBT | long_carry_reversion_watch | -0.907677 | -0.002554 | -0.001505 | 0.870140 | 0.002102 | 5.884807 | long perp receives funding and mark is below oracle |
| kNEIRO | long_carry_reversion_watch | -1.007411 | -0.001717 | -0.000780 | 0.390791 | 0.002501 | 5.839589 | long perp receives funding and mark is below oracle |

## Interpretation

`long_carry_reversion_watch` means long perp receives funding while the mark is below oracle. `short_carry_reversion_watch` means short perp receives funding while the mark is above oracle. OI/volume is a crowding proxy, not proof of forced liquidations or future returns.
