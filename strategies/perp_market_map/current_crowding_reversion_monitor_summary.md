# Current Crowding Reversion Monitor

This repeats the current crowding/reversion screen over a short window. It is a persistence check, not a trade instruction.

| asset | action | obs | mean score | min score | mean funding | min abs funding | mean mark/oracle | mean OI/volume | mean impact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZRO | short_carry_reversion_watch | 6 | 12.024626 | 11.985700 | 0.312897 | 0.311987 | 0.000802 | 12.227323 | 0.000509 |
| HEMI | short_carry_reversion_watch | 6 | 10.389136 | 10.357548 | 0.109500 | 0.109500 | 0.000781 | 19.604372 | 0.002310 |
| APEX | short_carry_reversion_watch | 6 | 10.097814 | 10.073411 | 0.109500 | 0.109500 | 0.001348 | 14.364052 | 0.005779 |
| HMSTR | short_carry_reversion_watch | 6 | 10.019696 | 9.996302 | 0.109500 | 0.109500 | 0.005469 | 24.418572 | 0.010870 |
| VIRTUAL | long_carry_reversion_watch | 6 | 10.005157 | 9.971207 | -0.749199 | 0.748017 | -0.001671 | 5.366831 | 0.000815 |
| AAVE | long_carry_reversion_watch | 6 | 9.507456 | 9.482472 | -0.080156 | 0.078114 | -0.001010 | 8.901500 | 0.000490 |
| DYDX | short_carry_reversion_watch | 6 | 9.425403 | 9.258347 | 0.109500 | 0.109500 | 0.001154 | 9.260286 | 0.005706 |
| STBL | short_carry_reversion_watch | 6 | 9.297349 | 9.248105 | 0.109500 | 0.109500 | 0.001255 | 8.928380 | 0.003474 |
| PURR | short_carry_reversion_watch | 6 | 8.948470 | 8.915940 | 0.109500 | 0.109500 | 0.004387 | 8.821960 | 0.009577 |
| SNX | long_carry_reversion_watch | 6 | 8.935664 | 8.847252 | -0.988616 | 0.981046 | -0.001976 | 3.650015 | 0.002145 |
| TRUMP | long_carry_reversion_watch | 6 | 8.568777 | 8.540771 | -0.550379 | 0.544885 | -0.001064 | 5.182249 | 0.000650 |
| XMR | short_carry_reversion_watch | 6 | 8.304716 | 8.270404 | 0.521475 | 0.515556 | 0.001283 | 4.603435 | 0.000841 |
| AZTEC | short_carry_reversion_watch | 6 | 8.266927 | 8.230295 | 0.109500 | 0.109500 | 0.000105 | 7.953498 | 0.002741 |
| ATOM | long_carry_reversion_watch | 6 | 8.061450 | 8.033745 | -0.305025 | 0.304294 | -0.001624 | 6.383318 | 0.002031 |
| CFX | short_carry_reversion_watch | 6 | 7.718077 | 7.657049 | 0.444276 | 0.436983 | 0.002080 | 5.088784 | 0.001614 |
| UMA | long_carry_reversion_watch | 6 | 7.597730 | 7.469086 | -1.006985 | 0.997776 | -0.003416 | 2.668377 | 0.002681 |
| SEI | long_carry_reversion_watch | 6 | 7.317391 | 7.298786 | -0.709193 | 0.708384 | -0.001705 | 3.045498 | 0.000978 |
| MET | long_carry_reversion_watch | 6 | 7.106944 | 7.090580 | -0.042223 | 0.040054 | -0.001461 | 6.968383 | 0.002374 |
| BSV | long_carry_reversion_watch | 6 | 6.930059 | 6.906051 | -0.829176 | 0.824063 | -0.002557 | 2.580588 | 0.003265 |
| GRIFFAIN | short_carry_reversion_watch | 6 | 6.832401 | 6.782004 | 0.109500 | 0.109500 | 0.001029 | 6.517229 | 0.003249 |

## Interpretation

Rows that appear in every sample are persistence candidates. They still need future-return labels, funding-decay labels, and execution-cost checks before becoming strategy inputs.
