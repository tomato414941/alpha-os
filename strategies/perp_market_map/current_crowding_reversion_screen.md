# Current Crowding Reversion Screen

This screen looks for perp states where funding carry and mark/oracle reversion point in the same direction. It is not a trade instruction.

| asset | action | annualized funding | mark/oracle | premium | OI/volume | impact spread | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| STABLE | long_carry_reversion_watch | -5.717735 | -0.006950 | -0.005858 | 2.782264 | 0.002415 | 37.107633 | long perp receives funding and mark is below oracle |
| LAYER | long_carry_reversion_watch | -5.169090 | -0.005402 | -0.003423 | 2.263612 | 0.003907 | 30.741807 | long perp receives funding and mark is below oracle |
| MOVE | long_carry_reversion_watch | -1.735768 | -0.003479 | -0.001491 | 0.406375 | 0.003206 | 11.084848 | long perp receives funding and mark is below oracle |
| HEMI | short_carry_reversion_watch | 0.195030 | 0.002321 | 0.000536 | 19.534814 | 0.003919 | 10.812404 | short perp receives funding and mark is above oracle |
| APEX | short_carry_reversion_watch | 0.109500 | 0.001740 | 0.000155 | 16.063808 | 0.003656 | 10.347214 | short perp receives funding and mark is above oracle |
| ZRO | short_carry_reversion_watch | 0.109500 | 0.000091 | 0.000080 | 9.407878 | 0.000456 | 10.076349 | short perp receives funding and mark is above oracle |
| STBL | short_carry_reversion_watch | 0.305599 | 0.001570 | 0.000157 | 8.002896 | 0.002313 | 9.601176 | short perp receives funding and mark is above oracle |
| TRX | long_carry_reversion_watch | -0.160896 | -0.000704 | -0.000450 | 8.499026 | 0.000386 | 9.549385 | long perp receives funding and mark is below oracle |
| ATOM | long_carry_reversion_watch | -0.143536 | -0.001860 | -0.000958 | 8.188473 | 0.002034 | 8.984306 | long perp receives funding and mark is below oracle |
| ETC | long_carry_reversion_watch | -0.254855 | -0.001127 | -0.000793 | 7.578181 | 0.001267 | 8.920581 | long perp receives funding and mark is below oracle |
| CC | short_carry_reversion_watch | 0.109500 | 0.000641 | 0.000000 | 7.994982 | 0.003588 | 8.343841 | short perp receives funding and mark is above oracle |
| VINE | short_carry_reversion_watch | 0.109500 | 0.000181 | 0.000000 | 7.910171 | 0.004520 | 7.991332 | short perp receives funding and mark is above oracle |
| GRASS | short_carry_reversion_watch | 0.109500 | 0.000496 | 0.000000 | 7.456545 | 0.003882 | 7.758382 | short perp receives funding and mark is above oracle |
| BIO | long_carry_reversion_watch | -0.970757 | -0.002088 | -0.000280 | 1.603186 | 0.001716 | 7.608759 | long perp receives funding and mark is below oracle |
| TRUMP | long_carry_reversion_watch | -0.651636 | -0.001645 | -0.001140 | 3.226568 | 0.000559 | 7.437857 | long perp receives funding and mark is below oracle |
| S | long_carry_reversion_watch | -0.795462 | -0.002607 | -0.001551 | 2.756154 | 0.001720 | 7.091106 | long perp receives funding and mark is below oracle |
| PURR | short_carry_reversion_watch | 0.109500 | 0.001380 | 0.000000 | 7.979818 | 0.018111 | 6.961853 | short perp receives funding and mark is above oracle |
| XMR | short_carry_reversion_watch | 0.123733 | 0.001042 | 0.000256 | 5.098442 | 0.000984 | 5.965970 | short perp receives funding and mark is above oracle |
| PROVE | long_carry_reversion_watch | -0.917752 | -0.001379 | -0.000944 | 0.628974 | 0.002147 | 5.799003 | long perp receives funding and mark is below oracle |
| VIRTUAL | long_carry_reversion_watch | -0.381341 | -0.001316 | -0.000894 | 3.110605 | 0.000844 | 5.572181 | long perp receives funding and mark is below oracle |
| CFX | short_carry_reversion_watch | 0.109500 | 0.000129 | 0.000000 | 4.979655 | 0.003480 | 5.282538 | short perp receives funding and mark is above oracle |
| MINA | long_carry_reversion_watch | -0.437652 | -0.001740 | -0.000723 | 3.032541 | 0.002490 | 5.222508 | long perp receives funding and mark is below oracle |
| HYPER | long_carry_reversion_watch | -0.843245 | -0.002191 | -0.001555 | 0.558779 | 0.001941 | 5.149935 | long perp receives funding and mark is below oracle |
| INJ | long_carry_reversion_watch | -0.537202 | -0.001319 | -0.000891 | 1.292574 | 0.000984 | 4.898057 | long perp receives funding and mark is below oracle |
| NXPC | long_carry_reversion_watch | -0.361729 | -0.001910 | -0.000890 | 2.623945 | 0.001699 | 4.644201 | long perp receives funding and mark is below oracle |

## Interpretation

`long_carry_reversion_watch` means long perp receives funding while the mark is below oracle. `short_carry_reversion_watch` means short perp receives funding while the mark is above oracle. OI/volume is a crowding proxy, not proof of forced liquidations or future returns.
