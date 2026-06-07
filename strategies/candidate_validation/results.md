# Candidate Validation Results

Data:

- source: current strategy candidate CSVs
- market data: Hyperliquid public candle snapshots
- output: recent return and volume context for active candidates
- forward label: elapsed monitor samples joined to subsequent Hyperliquid candle returns

Run:

```bash
uv run python -m strategies.candidate_validation.current_hl_candidate_return_context
uv run python -m strategies.candidate_validation.current_hl_signal_forward_labels
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

## Current HL Signal Forward Labels

This labels elapsed monitor samples with subsequent Hyperliquid candle returns.
It is still a small forward-label check, not a final alpha test.

| source | action | asset | obs | cov15 | cov1h | mean 15m | mean 1h | hit15 | hit1h |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| okx_hl_current | paper_24h_monitor | WLD | 10 | 10 | 0 | 0.019682 |  | 1.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | MEGA | 6 | 6 | 0 | 0.017831 |  | 1.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | IP | 6 | 6 | 0 | 0.015990 |  | 1.000000 |  |
| perp_carry_reversion | short_carry_reversion_watch | XMR | 6 | 6 | 0 | 0.011059 |  | 1.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | ZORA | 6 | 6 | 0 | 0.005486 |  | 1.000000 |  |
| okx_hl_current | paper_24h_monitor | STABLE | 12 | 12 | 0 | -0.001692 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | AERO | 6 | 6 | 0 | -0.001786 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | MON | 6 | 6 | 0 | -0.002903 |  | 0.000000 |  |

Interpretation:

- `WLD`, `MEGA`, `IP`, and `XMR` have positive elapsed 15m forward price labels.
- `STABLE` remains interesting for funding dislocation, but its elapsed 15m
  price-only label is negative. This label excludes funding PnL, hedge PnL,
  fees, and execution effects.
- `AERO` and `MON` were structurally interesting candidates, but this short
  price label does not currently support them.
- 1h labels have zero coverage because enough time had not elapsed for these
  sample timestamps when this was run.
