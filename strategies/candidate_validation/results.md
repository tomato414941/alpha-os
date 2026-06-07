# Candidate Validation Results

Data:

- source: current strategy candidate CSVs
- market data: Hyperliquid public candle snapshots
- output: recent return and volume context for active candidates

Run:

```bash
uv run python -m strategies.candidate_validation.current_hl_candidate_return_context
```

This is not a causal alpha test. It keeps current candidates connected to
realized market behavior so screens do not stay detached from price and volume.

## Current HL Candidate Return Context

| symbol | sources | close | 1h | 4h | 24h | vol24h | action | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| WLD | cross_exchange_funding | 0.49185000 | 0.105355 | 0.171267 | 0.177943 | 121923655.90 | single_source_momentum_context | 34.098874 |
| MEGA | perp_carry_reversion | 0.04627600 | 0.035118 | 0.040261 | 0.063792 | 24245736.00 | single_source_momentum_context | 20.524871 |
| STABLE | cross_exchange_funding;perp_carry_reversion | 0.03366900 | -0.003463 | 0.001845 | 0.036256 | 35488188.00 | multi_source_watch | 20.438540 |
| MON | perp_carry_reversion | 0.02226100 | -0.002822 | -0.023426 | 0.061817 | 111443620.00 | single_source_context | 16.453517 |
| BABY | perp_carry_reversion | 0.01556600 | 0.011699 | -0.001668 | 0.045259 | 107875950.00 | single_source_context | 16.253271 |
| AERO | attention_market_join;perp_carry_reversion | 0.32953000 | 0.006844 | -0.000879 | 0.040544 | 1217829.00 | multi_source_watch | 11.946201 |

Interpretation:

- `WLD` has the strongest recent realized move among active Hyperliquid
  candidates, but it is still single-source.
- `STABLE` and `AERO` are more structurally interesting because they appear in
  multiple research lanes.
- The next stronger test is forward labeling from signal timestamps, not only
  recent return context.
