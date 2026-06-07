# Perp Market Map Results

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.perp_market_map.current_hyperliquid_snapshot
uv run python -m strategies.perp_market_map.current_crowding_reversion_screen
uv run python -m strategies.perp_market_map.current_crowding_reversion_monitor --samples 6 --delay-seconds 10
uv run python -m strategies.perp_market_map.current_okx_perp_pressure
```

Interpretation:

- positive funding means short perp receives funding
- negative funding means long perp receives funding
- high open interest and volume improve feasibility
- wide impact spread weakens feasibility
- this is a current snapshot, not a historical backtest

## Top Snapshot Rows

| asset | annualized funding | OI notional | 24h notional volume | impact spread | carry side |
| --- | ---: | ---: | ---: | ---: | --- |
| SAGA | -1.1683 | 133588 | 188711 | 0.002191 | long perp receives funding |
| BABY | -0.9766 | 599169 | 1740757 | 0.001804 | long perp receives funding |
| kNEIRO | -1.0074 | 114590 | 293226 | 0.002501 | long perp receives funding |
| AIXBT | -0.9077 | 236129 | 271369 | 0.002102 | long perp receives funding |
| UMA | -1.0280 | 157812 | 71670 | 0.003938 | long perp receives funding |

The immediate next question is not whether these rows are profitable today. The
right question is whether large funding, premium, open interest, and impact
spread states persist long enough to execute and hedge.

## Current Crowding Reversion Screen

This screen looks for perp states where funding carry and mark/oracle reversion
point in the same direction.

| asset | action | annualized funding | mark/oracle | premium | OI/volume | impact spread | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MON | long_carry_reversion_watch | -0.739466 | -0.001966 | -0.001341 | 11.538740 | 0.001298 | 14.821547 |
| AERO | long_carry_reversion_watch | -0.288468 | -0.001183 | -0.000549 | 12.183732 | 0.001646 | 11.585710 |
| XAI | long_carry_reversion_watch | -0.296142 | -0.001266 | 0.000000 | 13.649106 | 0.005070 | 10.939232 |
| ZRO | short_carry_reversion_watch | 0.109500 | 0.000545 | 0.000589 | 10.302630 | 0.000400 | 10.728549 |
| HEMI | short_carry_reversion_watch | 0.109500 | 0.001079 | 0.001079 | 15.345254 | 0.002870 | 10.397096 |
| STBL | short_carry_reversion_watch | 0.109500 | 0.000565 | 0.000000 | 9.740925 | 0.004681 | 9.915925 |
| PURR | short_carry_reversion_watch | 0.109500 | 0.002849 | 0.001274 | 8.719279 | 0.009957 | 8.685959 |
| MORPHO | long_carry_reversion_watch | -0.696141 | -0.002162 | -0.001128 | 4.355470 | 0.001276 | 8.583529 |

Interpretation:

- `long_carry_reversion_watch` means long perp receives funding while mark is
  below oracle.
- `short_carry_reversion_watch` means short perp receives funding while mark is
  above oracle.
- `OI/volume` is computed from open-interest notional divided by 24h notional
  volume. It is a crowding proxy, not proof of forced liquidations.
- The next validation is repeated monitoring plus labels for subsequent return,
  funding decay, and execution cost.

## Current Crowding Reversion Monitor

This repeats the crowding/reversion screen over a short window.

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

Interpretation:

- `MON/AERO/ZRO/HEMI/PURR/MORPHO/SNX/IP` persisted in every sample.
- This is broader than the STABLE cross-exchange funding candidate: it gives a
  separate perp-market state watchlist.
- The missing work is still large: future-return labels, liquidation/funding
  event labels, execution costs, and whether these states decay before entry.

## Current OKX Perp Pressure

This maps current OKX USDT swap funding, premium, open interest, volume, and
near-touch spread. It is a separate venue screen from Hyperliquid.

| asset | action | ann funding | settled ann funding | premium | OI USD | volume USD | OI/vol | spread bps | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HOME | long_carry_discount_watch | -10.950000 | -10.950000 | -0.078950 | 2926534 | 56649695 | 0.0517 | 6.3032 | 1646.868988 |
| EDEN | long_carry_discount_watch | -5.114332 | -5.775057 | -0.007826 | 1679073 | 43635151 | 0.0385 | 3.8767 | 729.651254 |
| MU | short_carry_watch | 1.598865 | 0.248700 | -0.003762 | 32729725 | 67917339 | 0.4819 | 0.4443 | 271.103687 |
| DRAM | short_carry_premium_watch | 2.039144 | 0.285458 | 0.001569 | 5140502 | 8773923 | 0.5859 | 1.7052 | 169.551114 |
| QQQ | short_carry_premium_watch | 1.321647 | 0.419271 | 0.001674 | 7829009 | 6260009 | 1.2506 | 0.1416 | 113.748315 |
| ZEC | long_carry_discount_watch | -0.672485 | -0.230588 | -0.001023 | 59977408 | 1080594206 | 0.0555 | 0.2320 | 71.416463 |
| MON | long_carry_discount_watch | -0.532534 | -0.496308 | -0.001211 | 2574319 | 6708290 | 0.3838 | 4.5239 | 37.373778 |
| WLD | long_carry_discount_watch | -0.545352 | -0.501797 | -0.000039 | 37991265 | 466161398 | 0.0815 | 2.0490 | 36.508262 |
| IP | long_carry_discount_watch | -0.315960 | -0.103836 | -0.002279 | 3981302 | 13717867 | 0.2902 | 3.1392 | 31.810813 |

Interpretation:

- `HOME` and `EDEN` are extreme current funding/premium pressure rows, but
  they need fast validation because extreme funding can decay before entry.
- `WLD`, `MON`, and `IP` overlap with existing Hyperliquid/candidate-validation
  work, which makes them useful cross-venue follow-up candidates.
- This screen still lacks forward labels, funding decay labels, actual fees,
  maker/taker fill probability, and liquidation data.
