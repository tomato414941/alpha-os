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
uv run python -m strategies.candidate_validation.current_cross_lane_candidate_review
uv run python -m strategies.candidate_validation.current_signal_family_review
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
| okx_hl_current | paper_24h_monitor | WLD | 10 | 10 | 10 | 0.019682 | 0.072090 | 1.000000 | 1.000000 |
| perp_carry_reversion | long_carry_reversion_watch | MEGA | 6 | 6 | 0 | 0.017831 |  | 1.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | IP | 6 | 6 | 0 | 0.015990 |  | 1.000000 |  |
| perp_carry_reversion | short_carry_reversion_watch | XMR | 6 | 6 | 0 | 0.011059 |  | 1.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | ZORA | 6 | 6 | 0 | 0.005486 |  | 1.000000 |  |
| okx_hl_current | paper_24h_monitor | STABLE | 12 | 12 | 12 | -0.001692 | 0.000119 | 0.000000 | 1.000000 |
| perp_carry_reversion | long_carry_reversion_watch | AERO | 6 | 6 | 0 | -0.001786 |  | 0.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | MON | 6 | 6 | 0 | -0.002903 |  | 0.000000 |  |

Interpretation:

- `WLD`, `MEGA`, `IP`, and `XMR` have positive elapsed 15m forward price labels.
- `WLD` also has a positive elapsed 1h price label.
- `STABLE` remains interesting for funding dislocation, but its elapsed 15m
  price-only label is negative. This label excludes funding PnL, hedge PnL,
  fees, and execution effects.
- `AERO` and `MON` were structurally interesting candidates, but this short
  price label does not currently support them.
- 1h labels are currently available only for the older OKX/HL monitor samples.

## Current Cross-Lane Candidate Review

This consolidates current candidate screens and first short-horizon labels. It
is a triage board, not a deployable strategy ranking.

| asset | score | lanes | positive labels | negative labels | pending labels | note |
| --- | ---: | --- | --- | --- | --- | --- |
| WLD | 7.0571 | hl_candidate_label; okx_pressure; okx_liquidation | hl15=0.0197; okx_pressure15=0.0247; liq_cont15=0.0273 |  |  | first labels support follow-up |
| MEGA | 2.8916 | hl_candidate_label | hl15=0.0178 |  |  | first labels support follow-up |
| IP | 2.8166 | hl_candidate_label; okx_pressure | hl15=0.0160 | okx_pressure15=-0.0009 |  | mixed evidence; isolate which source is real |
| ALLO | 2.5965 | okx_pressure; okx_liquidation | liq_cont15=0.0198 | okx_pressure15=-0.0078 |  | mixed evidence; isolate which source is real |
| XMR | 2.5530 | hl_candidate_label | hl15=0.0111 |  |  | first labels support follow-up |
| HOME | 2.3356 | okx_pressure; okx_liquidation | okx_pressure15=0.0070 | liq_cont15=-0.0074 |  | mixed evidence; isolate which source is real |
| H | 2.2846 | okx_pressure; okx_liquidation | liq_cont15=0.0131 | okx_pressure15=-0.0005 |  | mixed evidence; isolate which source is real |
| ZORA | 2.2743 | hl_candidate_label | hl15=0.0055 |  |  | first labels support follow-up |
| ZRO | 2.0173 | hl_candidate_label; okx_pressure | hl15=0.0013 | okx_pressure15=-0.0011 |  | mixed evidence; isolate which source is real |

Interpretation:

- `WLD` is now the cleanest current follow-up because HL forward label, OKX
  pressure label, and OKX liquidation continuation label all support it.
- `IP` is still important, but the latest OKX pressure label is now negative.
- `HOME`, `EDEN`, and `ALLO` remain interesting, but they are source-specific
  and currently mixed.

## Current Signal Family Review

This aggregates short-horizon labels by signal family. It asks which kind of
signal is currently showing support, not only which asset is on top.

| family | obs | cov15 | mean15 | hit15 | max15 | min15 | score | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| okx_liquidation:short_liquidation_squeeze_watch | 17 | 17 | 0.004270 | 0.882353 | 0.027309 | -0.009277 | 1.387726 | supported by first labels |
| okx_pressure:long_carry_discount_watch | 32 | 32 | 0.001608 | 0.750000 | 0.024699 | -0.008581 | 0.401907 | supported by first labels |
| hl_candidate:okx_hl_current:paper_24h_monitor | 22 | 22 | 0.008023 | 0.454545 | 0.019682 | -0.001692 | 0.000000 | positive mean but weak hit rate |
| hl_candidate:perp_carry_reversion:long_carry_reversion_watch | 120 | 120 | 0.000472 | 0.266667 | 0.017831 | -0.007837 | 0.000000 | positive mean but weak hit rate |
| hl_candidate:perp_carry_reversion:short_carry_reversion_watch | 30 | 30 | 0.000858 | 0.433333 | 0.011059 | -0.005924 | 0.000000 | positive mean but weak hit rate |
| okx_pressure:short_carry_watch | 45 | 45 | -0.001622 | 0.200000 | 0.008225 | -0.010693 | 0.000000 | not supported by first labels |
| okx_pressure:short_carry_premium_watch | 13 | 13 | -0.002705 | 0.230769 | 0.009967 | -0.013383 | 0.000000 | not supported by first labels |
| okx_liquidation:long_liquidation_cascade_watch | 8 | 8 | -0.001233 | 0.375000 | 0.002599 | -0.007351 | 0.000000 | not supported by first labels |

Interpretation:

- `short_liquidation_squeeze_watch` is the strongest current signal family:
  17 covered labels, 0.88 hit rate, and positive mean 15m continuation.
- `long_carry_discount_watch` also has initial support, but the average label
  is much smaller.
- The current short-side carry families are not supported by first labels.
