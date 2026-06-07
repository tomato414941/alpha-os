# Current OKX Perp Pressure

This maps current OKX USDT swap funding, premium, open interest, volume, and near-touch spread. It is a candidate screen, not a deployable strategy.

| asset | action | ann funding | settled ann funding | premium | OI USD | volume USD | OI/vol | spread bps | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HOME | long_carry_discount_watch | -10.950000 | -10.950000 | -0.078950 | 2926534 | 56649695 | 0.0517 | 6.3032 | 1646.868988 |
| EDEN | long_carry_discount_watch | -5.114332 | -5.775057 | -0.007826 | 1679073 | 43635151 | 0.0385 | 3.8767 | 729.651254 |
| MU | short_carry_watch | 1.598865 | 0.248700 | -0.003762 | 32729725 | 67917339 | 0.4819 | 0.4443 | 271.103687 |
| DRAM | short_carry_premium_watch | 2.039144 | 0.285458 | 0.001569 | 5140502 | 8773923 | 0.5859 | 1.7052 | 169.551114 |
| QQQ | short_carry_premium_watch | 1.321647 | 0.419271 | 0.001674 | 7829009 | 6260009 | 1.2506 | 0.1416 | 113.748315 |
| SNDK | short_carry_watch | 1.432290 | 0.000000 | 0.000000 | 11956041 | 21937098 | 0.5450 | 0.6349 | 74.412368 |
| ZEC | long_carry_discount_watch | -0.672485 | -0.230588 | -0.001023 | 59977408 | 1080594206 | 0.0555 | 0.2320 | 71.416463 |
| SAHARA | short_carry_premium_watch | 0.513770 | 0.335219 | 0.001436 | 6056433 | 7852052 | 0.7713 | 2.8674 | 41.246686 |
| SOXL | short_carry_premium_watch | 0.597672 | 0.768456 | 0.000858 | 2693483 | 9603091 | 0.2805 | 5.3206 | 38.287788 |
| MON | long_carry_discount_watch | -0.532534 | -0.496308 | -0.001211 | 2574319 | 6708290 | 0.3838 | 4.5239 | 37.373778 |
| SOL | long_carry_discount_watch | -0.409140 | -0.281216 | -0.000464 | 183114055 | 734192529 | 0.2494 | 1.5474 | 36.907196 |
| WLD | long_carry_discount_watch | -0.545352 | -0.501797 | -0.000039 | 37991265 | 466161398 | 0.0815 | 2.0490 | 36.508262 |
| BABY | long_carry_discount_watch | -0.403826 | -0.818203 | -0.001896 | 1726431 | 21211148 | 0.0814 | 6.4371 | 35.884301 |
| BEAT | short_carry_premium_watch | 0.258316 | 0.811695 | 0.001946 | 27119409 | 277896650 | 0.0976 | 0.3594 | 31.987275 |
| IP | long_carry_discount_watch | -0.315960 | -0.103836 | -0.002279 | 3981302 | 13717867 | 0.2902 | 3.1392 | 31.810813 |
| OPN | long_carry_discount_watch | -0.534079 | -0.441853 | -0.000051 | 3355961 | 65866288 | 0.0510 | 7.6894 | 27.869419 |
| TRX | long_carry_discount_watch | -0.333278 | -0.114154 | -0.000913 | 16576296 | 16232539 | 1.0212 | 3.0474 | 25.240281 |
| BSB | short_carry_premium_watch | 0.174609 | 0.054750 | 0.002726 | 8532438 | 192049265 | 0.0444 | 3.0400 | 23.658963 |
| RAVE | short_carry_premium_watch | 0.287839 | 0.439024 | 0.001318 | 3449398 | 7703216 | 0.4478 | 2.9660 | 21.471152 |
| CL | short_carry_premium_watch | 0.222413 | 0.234956 | 0.001454 | 26011391 | 21260174 | 1.2235 | 1.0858 | 20.859757 |
| MRVL | short_carry_watch | 0.109500 | -0.025581 | -0.003662 | 9733686 | 37268210 | 0.2612 | 3.4513 | 16.368314 |
| NVDA | short_carry_premium_watch | 0.221716 | 0.260287 | 0.000918 | 10917082 | 13152345 | 0.8300 | 0.4828 | 16.204607 |
| BILL | short_carry_premium_watch | 0.211133 | 0.054750 | 0.001207 | 3600118 | 13304113 | 0.2706 | 2.5332 | 15.785746 |
| ADA | long_carry_discount_watch | -0.209088 | -0.097469 | -0.000618 | 20895401 | 68038811 | 0.3071 | 6.1824 | 15.628838 |
| ETH | long_carry_discount_watch | -0.131919 | -0.153901 | -0.000412 | 1222291873 | 6993465604 | 0.1748 | 0.0616 | 14.233383 |

## Interpretation

High scores mean the instrument is liquid enough to inspect and has large current funding or premium pressure. This does not include future-return labels, funding decay labels, fees, maker/taker fill probability, or liquidation data.
