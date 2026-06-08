# Current Crowding Reversion Screen

This screen looks for perp states where funding carry and mark/oracle reversion point in the same direction. It is not a trade instruction.

| asset | action | annualized funding | mark/oracle | premium | OI/volume | impact spread | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LAYER | long_carry_reversion_watch | -18.393339 | -0.010895 | -0.007462 | 0.705071 | 0.004598 | 113.770421 | long perp receives funding and mark is below oracle |
| STABLE | long_carry_reversion_watch | -1.405638 | -0.003482 | -0.002143 | 4.787991 | 0.002030 | 12.968764 | long perp receives funding and mark is below oracle |
| HEMI | short_carry_reversion_watch | 0.485146 | 0.001812 | 0.000543 | 19.549388 | 0.002712 | 12.305137 | short perp receives funding and mark is above oracle |
| ATOM | long_carry_reversion_watch | -0.760028 | -0.002448 | -0.001446 | 7.601601 | 0.001952 | 11.897053 | long perp receives funding and mark is below oracle |
| TRX | long_carry_reversion_watch | -0.240218 | -0.000950 | -0.000846 | 9.469053 | 0.000414 | 11.035735 | long perp receives funding and mark is below oracle |
| HMSTR | long_carry_reversion_watch | -1.753693 | -0.006024 | 0.000000 | 0.650606 | 0.012121 | 10.802448 | long perp receives funding and mark is below oracle |
| APEX | short_carry_reversion_watch | 0.109500 | 0.000973 | 0.000000 | 16.791643 | 0.003496 | 10.280058 | short perp receives funding and mark is above oracle |
| S | long_carry_reversion_watch | -1.243906 | -0.002532 | -0.001546 | 3.352084 | 0.001352 | 9.978765 | long perp receives funding and mark is below oracle |
| XAI | long_carry_reversion_watch | -0.026360 | -0.002457 | -0.001229 | 21.129369 | 0.006165 | 9.772757 | long perp receives funding and mark is below oracle |
| ZRO | short_carry_reversion_watch | 0.109500 | 0.000902 | 0.000879 | 8.930122 | 0.000411 | 9.707751 | short perp receives funding and mark is above oracle |
| ZEC | long_carry_reversion_watch | -1.020960 | -0.001614 | -0.001239 | 0.657232 | 0.000422 | 9.495444 | long perp receives funding and mark is below oracle |
| IO | long_carry_reversion_watch | -1.416071 | -0.002404 | -0.001168 | 0.718330 | 0.002479 | 8.999405 | long perp receives funding and mark is below oracle |
| DYDX | short_carry_reversion_watch | 0.109500 | 0.000140 | 0.000000 | 8.437309 | 0.002798 | 8.796211 | short perp receives funding and mark is above oracle |
| BSV | long_carry_reversion_watch | -0.384118 | -0.002625 | -0.000435 | 6.569342 | 0.003561 | 8.374504 | long perp receives funding and mark is below oracle |
| STBL | short_carry_reversion_watch | 0.109500 | 0.000548 | 0.000000 | 8.093657 | 0.003759 | 8.370332 | short perp receives funding and mark is above oracle |
| ETC | long_carry_reversion_watch | -0.221444 | -0.001539 | -0.000657 | 7.203323 | 0.002046 | 8.334819 | long perp receives funding and mark is below oracle |
| AVNT | long_carry_reversion_watch | -1.134982 | -0.002680 | -0.001972 | 1.286847 | 0.001613 | 7.905881 | long perp receives funding and mark is below oracle |
| CFX | short_carry_reversion_watch | 0.109500 | 0.000779 | 0.000000 | 6.642085 | 0.002012 | 7.142697 | short perp receives funding and mark is above oracle |
| GRASS | short_carry_reversion_watch | 0.109500 | 0.000481 | 0.000000 | 6.595341 | 0.001611 | 7.127938 | short perp receives funding and mark is above oracle |
| AERO | long_carry_reversion_watch | -0.100289 | -0.001821 | -0.000847 | 6.472636 | 0.001413 | 7.126791 | long perp receives funding and mark is below oracle |
| PURR | short_carry_reversion_watch | 0.109500 | 0.002435 | 0.000000 | 7.213216 | 0.012950 | 6.819842 | short perp receives funding and mark is above oracle |
| MOVE | long_carry_reversion_watch | -1.016327 | -0.001724 | -0.001364 | 0.326315 | 0.001727 | 6.636620 | long perp receives funding and mark is below oracle |
| XMR | short_carry_reversion_watch | 0.109500 | 0.000379 | 0.000000 | 5.735461 | 0.001045 | 6.419946 | short perp receives funding and mark is above oracle |
| VIRTUAL | long_carry_reversion_watch | -0.380164 | -0.001345 | -0.001043 | 3.656549 | 0.000707 | 6.106941 | long perp receives funding and mark is below oracle |
| COMP | long_carry_reversion_watch | -0.371191 | -0.002311 | -0.001365 | 3.554699 | 0.002035 | 5.619428 | long perp receives funding and mark is below oracle |

## Interpretation

`long_carry_reversion_watch` means long perp receives funding while the mark is below oracle. `short_carry_reversion_watch` means short perp receives funding while the mark is above oracle. OI/volume is a crowding proxy, not proof of forced liquidations or future returns.
