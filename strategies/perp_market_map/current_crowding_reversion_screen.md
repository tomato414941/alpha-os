# Current Crowding Reversion Screen

This screen looks for perp states where funding carry and mark/oracle reversion point in the same direction. It is not a trade instruction.

| asset | action | annualized funding | mark/oracle | premium | OI/volume | impact spread | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| STABLE | long_carry_reversion_watch | -5.348045 | -0.006387 | -0.005033 | 2.370222 | 0.002458 | 34.895274 | long perp receives funding and mark is below oracle |
| PURR | short_carry_reversion_watch | 2.122974 | 0.004168 | 0.000000 | 9.117620 | 0.012025 | 20.829100 | short perp receives funding and mark is above oracle |
| ZRO | short_carry_reversion_watch | 0.668614 | 0.000929 | 0.000929 | 9.101651 | 0.000121 | 13.516294 | short perp receives funding and mark is above oracle |
| HEMI | short_carry_reversion_watch | 0.715667 | 0.003058 | 0.001079 | 18.741213 | 0.003587 | 13.502334 | short perp receives funding and mark is above oracle |
| AERO | long_carry_reversion_watch | -0.279082 | -0.002351 | -0.001566 | 9.835418 | 0.001513 | 11.553428 | long perp receives funding and mark is below oracle |
| XMR | short_carry_reversion_watch | 0.874185 | 0.001806 | 0.000929 | 4.522836 | 0.001446 | 10.653850 | short perp receives funding and mark is above oracle |
| APEX | short_carry_reversion_watch | 0.109500 | 0.000841 | 0.000000 | 15.511949 | 0.003382 | 10.282929 | short perp receives funding and mark is above oracle |
| VINE | short_carry_reversion_watch | 0.109500 | 0.002966 | 0.000000 | 9.578138 | 0.006288 | 9.753337 | short perp receives funding and mark is above oracle |
| CFX | short_carry_reversion_watch | 0.801201 | 0.002137 | 0.000218 | 4.283727 | 0.003177 | 8.877994 | short perp receives funding and mark is above oracle |
| 2Z | long_carry_reversion_watch | -0.723920 | -0.002152 | -0.001133 | 4.222672 | 0.001774 | 8.100529 | long perp receives funding and mark is below oracle |
| LDO | long_carry_reversion_watch | -0.017189 | -0.001102 | -0.000588 | 7.774878 | 0.001177 | 7.881510 | long perp receives funding and mark is below oracle |
| ZK | long_carry_reversion_watch | -0.983101 | -0.002203 | -0.001724 | 2.394277 | 0.002401 | 7.838401 | long perp receives funding and mark is below oracle |
| TRX | long_carry_reversion_watch | -0.661559 | -0.001256 | -0.001002 | 3.171247 | 0.000488 | 7.697001 | long perp receives funding and mark is below oracle |
| TRUMP | long_carry_reversion_watch | -0.483809 | -0.001384 | -0.001011 | 4.312852 | 0.000886 | 7.352398 | long perp receives funding and mark is below oracle |
| STBL | short_carry_reversion_watch | 0.109500 | 0.001548 | 0.000000 | 6.986897 | 0.003962 | 7.348878 | short perp receives funding and mark is above oracle |
| GRIFFAIN | short_carry_reversion_watch | 0.109500 | 0.000364 | 0.000000 | 5.982233 | 0.002670 | 6.293851 | short perp receives funding and mark is above oracle |
| MERL | long_carry_reversion_watch | -0.808167 | -0.002293 | -0.001268 | 1.859200 | 0.003327 | 6.028643 | long perp receives funding and mark is below oracle |
| SNX | long_carry_reversion_watch | -0.403180 | -0.002488 | -0.001631 | 3.669318 | 0.001799 | 5.932724 | long perp receives funding and mark is below oracle |
| REZ | short_carry_reversion_watch | 0.109500 | 0.000303 | 0.000000 | 5.641399 | 0.003937 | 5.833702 | short perp receives funding and mark is above oracle |
| ZEC | long_carry_reversion_watch | -0.606761 | -0.001249 | -0.000951 | 0.437189 | 0.000549 | 5.807721 | long perp receives funding and mark is below oracle |
| BABY | long_carry_reversion_watch | -0.868645 | -0.002151 | -0.001494 | 0.434269 | 0.002276 | 5.770176 | long perp receives funding and mark is below oracle |
| DYDX | short_carry_reversion_watch | 0.109500 | 0.000554 | 0.000000 | 5.180475 | 0.003393 | 5.544721 | short perp receives funding and mark is above oracle |
| POPCAT | long_carry_reversion_watch | -0.495940 | -0.001742 | -0.000943 | 2.713249 | 0.001647 | 5.536307 | long perp receives funding and mark is below oracle |
| TURBO | long_carry_reversion_watch | -0.379114 | -0.002294 | -0.002294 | 3.470140 | 0.001151 | 5.458689 | long perp receives funding and mark is below oracle |
| JTO | long_carry_reversion_watch | -0.664672 | -0.002211 | -0.001302 | 0.310879 | 0.001410 | 5.058383 | long perp receives funding and mark is below oracle |

## Interpretation

`long_carry_reversion_watch` means long perp receives funding while the mark is below oracle. `short_carry_reversion_watch` means short perp receives funding while the mark is above oracle. OI/volume is a crowding proxy, not proof of forced liquidations or future returns.
