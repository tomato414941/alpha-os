# Perp Market Map Results

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.perp_market_map.current_hyperliquid_snapshot
uv run python -m strategies.perp_market_map.current_crowding_reversion_screen
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
