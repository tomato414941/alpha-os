# Current Crowding Reversion Monitor

This repeats the current crowding/reversion screen over a short window. It is a persistence check, not a trade instruction.

| asset | action | obs | mean score | min score | mean funding | min abs funding | mean mark/oracle | mean OI/volume | mean impact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MON | long_carry_reversion_watch | 6 | 14.780596 | 14.755183 | -0.737254 | 0.735363 | -0.001838 | 11.455053 | 0.001402 |
| AERO | long_carry_reversion_watch | 6 | 11.427866 | 11.373155 | -0.259455 | 0.255132 | -0.001321 | 12.158566 | 0.001683 |
| ZRO | short_carry_reversion_watch | 6 | 10.681220 | 10.648295 | 0.109500 | 0.109500 | 0.000314 | 10.266260 | 0.000549 |
| HEMI | short_carry_reversion_watch | 6 | 10.466755 | 10.414978 | 0.109500 | 0.109500 | 0.001769 | 15.351004 | 0.002841 |
| PURR | short_carry_reversion_watch | 6 | 8.606307 | 8.580319 | 0.109500 | 0.109500 | 0.001350 | 8.692386 | 0.008952 |
| MORPHO | long_carry_reversion_watch | 6 | 8.568573 | 8.514109 | -0.704351 | 0.702762 | -0.002340 | 4.332427 | 0.001883 |
| SNX | long_carry_reversion_watch | 6 | 7.470613 | 7.445775 | -0.806949 | 0.802528 | -0.002062 | 3.194100 | 0.003041 |
| IP | long_carry_reversion_watch | 6 | 7.402554 | 7.346064 | -0.182005 | 0.180456 | -0.001531 | 6.347635 | 0.001287 |
| ATOM | long_carry_reversion_watch | 6 | 7.286934 | 7.275832 | -0.333333 | 0.329599 | -0.002163 | 5.367466 | 0.002109 |
| BABY | long_carry_reversion_watch | 6 | 6.904177 | 6.854977 | -1.030958 | 1.024513 | -0.002984 | 0.344470 | 0.002286 |
| MEGA | long_carry_reversion_watch | 6 | 6.890245 | 6.832217 | -0.100203 | 0.097941 | -0.001192 | 6.317092 | 0.001572 |
| SAGA | long_carry_reversion_watch | 6 | 6.587071 | 6.526121 | -1.123865 | 1.122951 | -0.001825 | 0.706852 | 0.002683 |
| STABLE | long_carry_reversion_watch | 6 | 6.468331 | 6.436906 | -0.724902 | 0.720835 | -0.002883 | 1.964420 | 0.002236 |
| UMA | long_carry_reversion_watch | 6 | 6.424840 | 6.323647 | -0.909964 | 0.898198 | -0.002009 | 2.198392 | 0.003995 |
| XMR | short_carry_reversion_watch | 6 | 6.393492 | 6.362630 | 0.109500 | 0.109500 | 0.000385 | 5.660685 | 0.000647 |
| BSV | long_carry_reversion_watch | 6 | 6.096481 | 6.062069 | -0.667734 | 0.665792 | -0.003328 | 2.532669 | 0.003394 |
| SEI | long_carry_reversion_watch | 6 | 6.077002 | 6.056716 | -0.585338 | 0.583031 | -0.001392 | 2.524968 | 0.001002 |
| BIO | long_carry_reversion_watch | 6 | 6.028059 | 5.939692 | -0.648077 | 0.643594 | -0.001424 | 2.249122 | 0.001432 |
| TRUMP | long_carry_reversion_watch | 6 | 5.914520 | 5.898974 | -0.175686 | 0.174818 | -0.000952 | 4.803859 | 0.000669 |
| ZORA | long_carry_reversion_watch | 6 | 5.887087 | 5.848803 | -0.454590 | 0.452366 | -0.002637 | 3.536016 | 0.003172 |

## Interpretation

Rows that appear in every sample are persistence candidates. They still need future-return labels, funding-decay labels, and execution-cost checks before becoming strategy inputs.
