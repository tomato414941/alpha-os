# Current Crowding Reversion Screen

This screen looks for perp states where funding carry and mark/oracle reversion point in the same direction. It is not a trade instruction.

| asset | action | annualized funding | mark/oracle | premium | OI/volume | impact spread | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZRO | short_carry_reversion_watch | 0.256010 | 0.000789 | 0.000900 | 12.252971 | 0.000455 | 11.672259 | short perp receives funding and mark is above oracle |
| HEMI | short_carry_reversion_watch | 0.378425 | 0.000899 | 0.000000 | 19.703356 | 0.003054 | 11.641575 | short perp receives funding and mark is above oracle |
| APEX | short_carry_reversion_watch | 0.109500 | 0.001489 | 0.000000 | 14.309105 | 0.004051 | 10.284631 | short perp receives funding and mark is above oracle |
| SNX | long_carry_reversion_watch | -1.210809 | -0.001733 | -0.000701 | 3.654131 | 0.002024 | 10.108747 | long perp receives funding and mark is below oracle |
| PAXG | long_carry_reversion_watch | -0.000747 | -0.000464 | -0.000406 | 10.739965 | 0.000110 | 10.050229 | long perp receives funding and mark is below oracle |
| HMSTR | short_carry_reversion_watch | 0.109500 | 0.005435 | 0.000000 | 24.624410 | 0.010811 | 9.999111 | short perp receives funding and mark is above oracle |
| VIRTUAL | long_carry_reversion_watch | -0.741187 | -0.001656 | -0.000854 | 5.365868 | 0.000611 | 9.966646 | long perp receives funding and mark is below oracle |
| DYDX | short_carry_reversion_watch | 0.109500 | 0.001404 | 0.000000 | 9.651768 | 0.007011 | 9.709236 | short perp receives funding and mark is above oracle |
| STBL | short_carry_reversion_watch | 0.109500 | 0.002828 | 0.000199 | 9.089647 | 0.004290 | 9.538847 | short perp receives funding and mark is above oracle |
| AAVE | long_carry_reversion_watch | -0.043629 | -0.001035 | -0.000253 | 8.908744 | 0.000590 | 9.251856 | long perp receives funding and mark is below oracle |
| TRUMP | long_carry_reversion_watch | -0.651719 | -0.001215 | -0.000759 | 5.188483 | 0.000784 | 9.191978 | long perp receives funding and mark is below oracle |
| PURR | short_carry_reversion_watch | 0.109500 | 0.003149 | 0.000000 | 8.824571 | 0.011027 | 8.682143 | short perp receives funding and mark is above oracle |
| XMR | short_carry_reversion_watch | 0.527437 | 0.001677 | 0.001968 | 4.656246 | 0.000135 | 8.522693 | short perp receives funding and mark is above oracle |
| BSV | long_carry_reversion_watch | -1.068790 | -0.003481 | -0.001194 | 2.583313 | 0.003475 | 8.281752 | long perp receives funding and mark is below oracle |
| ATOM | long_carry_reversion_watch | -0.218696 | -0.001240 | -0.000709 | 6.392065 | 0.001952 | 7.558009 | long perp receives funding and mark is below oracle |
| UMA | long_carry_reversion_watch | -0.990937 | -0.001829 | -0.000653 | 2.672487 | 0.002645 | 7.336723 | long perp receives funding and mark is below oracle |
| SEI | long_carry_reversion_watch | -0.714222 | -0.001623 | -0.001015 | 3.044581 | 0.001036 | 7.329539 | long perp receives funding and mark is below oracle |
| MET | long_carry_reversion_watch | -0.081762 | -0.001418 | -0.000092 | 6.986212 | 0.002491 | 7.305243 | long perp receives funding and mark is below oracle |
| SKY | long_carry_reversion_watch | -0.365613 | -0.001103 | 0.000000 | 5.309277 | 0.003136 | 7.125116 | long perp receives funding and mark is below oracle |
| CFX | short_carry_reversion_watch | 0.309965 | 0.001844 | 0.001289 | 5.116106 | 0.001685 | 6.941646 | short perp receives funding and mark is above oracle |
| GRIFFAIN | short_carry_reversion_watch | 0.109500 | 0.000493 | 0.000000 | 6.545632 | 0.003453 | 6.786741 | short perp receives funding and mark is above oracle |
| NIL | long_carry_reversion_watch | -0.832173 | -0.002776 | -0.001438 | 1.658474 | 0.002353 | 6.474081 | long perp receives funding and mark is below oracle |
| ETC | long_carry_reversion_watch | -0.012164 | -0.001179 | -0.000824 | 6.262677 | 0.001408 | 6.324999 | long perp receives funding and mark is below oracle |
| RSR | long_carry_reversion_watch | -0.041307 | -0.001438 | 0.000000 | 6.273255 | 0.005764 | 6.020115 | long perp receives funding and mark is below oracle |
| POPCAT | long_carry_reversion_watch | -0.493647 | -0.001775 | -0.000851 | 3.056060 | 0.001144 | 5.891072 | long perp receives funding and mark is below oracle |

## Interpretation

`long_carry_reversion_watch` means long perp receives funding while the mark is below oracle. `short_carry_reversion_watch` means short perp receives funding while the mark is above oracle. OI/volume is a crowding proxy, not proof of forced liquidations or future returns.
