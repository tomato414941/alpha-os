# Current Crowding Reversion Screen

This screen looks for perp states where funding carry and mark/oracle reversion point in the same direction. It is not a trade instruction.

| asset | action | annualized funding | mark/oracle | premium | OI/volume | impact spread | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| STABLE | long_carry_reversion_watch | -4.075735 | -0.005876 | -0.005054 | 2.852504 | 0.002298 | 27.308448 | long perp receives funding and mark is below oracle |
| MOVE | long_carry_reversion_watch | -2.877765 | -0.003109 | -0.001727 | 0.432102 | 0.002703 | 18.037914 | long perp receives funding and mark is below oracle |
| HEMI | short_carry_reversion_watch | 0.530788 | 0.001971 | 0.000358 | 19.508585 | 0.002505 | 12.564524 | short perp receives funding and mark is above oracle |
| CC | short_carry_reversion_watch | 0.481704 | 0.001708 | 0.000474 | 8.169580 | 0.002349 | 10.947415 | short perp receives funding and mark is above oracle |
| TRX | long_carry_reversion_watch | -0.254626 | -0.000888 | -0.000714 | 8.433910 | 0.000340 | 10.101472 | long perp receives funding and mark is below oracle |
| APEX | short_carry_reversion_watch | 0.109500 | 0.000464 | 0.000000 | 15.741726 | 0.005288 | 10.053361 | short perp receives funding and mark is above oracle |
| PURR | short_carry_reversion_watch | 0.109500 | 0.002041 | 0.001726 | 9.143000 | 0.003422 | 9.695175 | short perp receives funding and mark is above oracle |
| STBL | short_carry_reversion_watch | 0.364361 | 0.002616 | 0.000742 | 7.299858 | 0.003194 | 9.265368 | short perp receives funding and mark is above oracle |
| GRASS | short_carry_reversion_watch | 0.109500 | 0.000499 | 0.000000 | 7.220479 | 0.003796 | 7.532466 | short perp receives funding and mark is above oracle |
| ZEC | long_carry_reversion_watch | -0.778720 | -0.001493 | -0.000781 | 0.509608 | 0.000492 | 7.335110 | long perp receives funding and mark is below oracle |
| OP | long_carry_reversion_watch | -0.561795 | -0.001732 | -0.001732 | 3.724941 | 0.001123 | 7.161544 | long perp receives funding and mark is below oracle |
| TRUMP | long_carry_reversion_watch | -0.535037 | -0.001426 | -0.001075 | 3.220719 | 0.000637 | 6.670309 | long perp receives funding and mark is below oracle |
| UMA | long_carry_reversion_watch | -0.558313 | -0.002258 | -0.000924 | 4.135599 | 0.002932 | 6.668792 | long perp receives funding and mark is below oracle |
| COMP | long_carry_reversion_watch | -0.320252 | -0.002116 | -0.000829 | 4.891120 | 0.002215 | 6.580224 | long perp receives funding and mark is below oracle |
| REZ | short_carry_reversion_watch | 0.109500 | 0.000303 | 0.000000 | 6.273825 | 0.003939 | 6.458083 | short perp receives funding and mark is above oracle |
| PROVE | long_carry_reversion_watch | -0.976362 | -0.002135 | -0.001345 | 0.619940 | 0.001690 | 6.258000 | long perp receives funding and mark is below oracle |
| ME | long_carry_reversion_watch | -0.953190 | -0.003040 | -0.001920 | 0.686744 | 0.002408 | 6.016973 | long perp receives funding and mark is below oracle |
| TAO | long_carry_reversion_watch | -0.413561 | -0.001306 | -0.001017 | 2.904368 | 0.000729 | 5.832444 | long perp receives funding and mark is below oracle |
| XMR | short_carry_reversion_watch | 0.139179 | 0.000538 | 0.000114 | 4.784664 | 0.000842 | 5.721994 | short perp receives funding and mark is above oracle |
| W | long_carry_reversion_watch | -0.407464 | -0.002086 | -0.001043 | 3.465003 | 0.003135 | 5.560319 | long perp receives funding and mark is below oracle |
| CHIP | long_carry_reversion_watch | -0.006166 | -0.000798 | -0.000276 | 5.447229 | 0.001105 | 5.458586 | long perp receives funding and mark is below oracle |
| CFX | short_carry_reversion_watch | 0.109500 | 0.000495 | 0.000000 | 4.641092 | 0.002132 | 5.118662 | short perp receives funding and mark is above oracle |
| POL | long_carry_reversion_watch | -0.242313 | -0.001314 | -0.000880 | 3.635528 | 0.001130 | 5.072864 | long perp receives funding and mark is below oracle |
| MINA | long_carry_reversion_watch | -0.253655 | -0.001988 | -0.001234 | 3.017609 | 0.002703 | 4.277666 | long perp receives funding and mark is below oracle |
| GRIFFAIN | short_carry_reversion_watch | 0.109500 | 0.000941 | 0.000000 | 3.871791 | 0.003410 | 4.189690 | short perp receives funding and mark is above oracle |

## Interpretation

`long_carry_reversion_watch` means long perp receives funding while the mark is below oracle. `short_carry_reversion_watch` means short perp receives funding while the mark is above oracle. OI/volume is a crowding proxy, not proof of forced liquidations or future returns.
