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

## Current Cross-Lane Candidate Review

This consolidates current candidate screens and first short-horizon labels. It
is a triage board, not a deployable strategy ranking.

| asset | score | lanes | positive labels | negative labels | pending labels | note |
| --- | ---: | --- | --- | --- | --- | --- |
| IP | 3.0182 | hl_candidate_label; okx_pressure | hl15=0.0160; okx_pressure15=0.0016 |  |  | first labels support follow-up |
| MEGA | 2.8916 | hl_candidate_label | hl15=0.0178 |  |  | first labels support follow-up |
| WLD | 2.8362 | hl_candidate_label; okx_pressure; okx_liquidation | hl15=0.0197 | okx_pressure15=-0.0028; liq_cont15=-0.0016 |  | mixed evidence; isolate which source is real |
| EDEN | 2.7546 | okx_pressure; okx_liquidation | liq_cont15=0.0057 | okx_pressure15=-0.0055 |  | mixed evidence; isolate which source is real |
| ALLO | 2.7293 | okx_pressure; okx_liquidation | liq_cont15=0.0198 | okx_pressure15=-0.0052 |  | mixed evidence; isolate which source is real |
| XMR | 2.5530 | hl_candidate_label | hl15=0.0111 |  |  | first labels support follow-up |
| H | 2.3700 | okx_pressure; okx_liquidation | okx_pressure15=0.0006; liq_cont15=0.0131 |  |  | first labels support follow-up |
| ZORA | 2.2743 | hl_candidate_label | hl15=0.0055 |  |  | first labels support follow-up |
| HOME | 2.1918 | okx_pressure; okx_liquidation | okx_pressure15=0.0035 | liq_cont15=-0.0032 |  | mixed evidence; isolate which source is real |
| ZRO | 2.1842 | hl_candidate_label; okx_pressure | hl15=0.0013; okx_pressure15=0.0011 |  |  | first labels support follow-up |

Interpretation:

- `IP` is the cleanest current follow-up because both HL forward label and OKX
  pressure label support it.
- `WLD` is important but mixed: it has the strongest HL candidate label while
  latest OKX pressure and liquidation labels are negative.
- `HOME`, `EDEN`, and `ALLO` remain interesting, but they are source-specific
  and currently mixed.

## Current Signal Family Review

This aggregates short-horizon labels by signal family. It asks which kind of
signal is currently showing support, not only which asset is on top.

| family | obs | cov15 | mean15 | hit15 | max15 | min15 | score | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| okx_liquidation:short_liquidation_squeeze_watch | 17 | 17 | 0.003543 | 0.882353 | 0.019775 | -0.001606 | 1.151348 | supported by first labels |
| okx_pressure:long_carry_discount_watch | 32 | 32 | 0.000710 | 0.718750 | 0.007264 | -0.005525 | 0.155283 | supported by first labels |
| hl_candidate:okx_hl_current:paper_24h_monitor | 22 | 22 | 0.008023 | 0.454545 | 0.019682 | -0.001692 | 0.000000 | positive mean but weak hit rate |
| hl_candidate:perp_carry_reversion:long_carry_reversion_watch | 120 | 114 | 0.000497 | 0.280702 | 0.017831 | -0.007837 | 0.000000 | positive mean but weak hit rate |
| hl_candidate:perp_carry_reversion:short_carry_reversion_watch | 30 | 30 | 0.000858 | 0.433333 | 0.011059 | -0.005924 | 0.000000 | positive mean but weak hit rate |
| okx_pressure:short_carry_watch | 45 | 45 | -0.000925 | 0.266667 | 0.002930 | -0.011341 | 0.000000 | not supported by first labels |
| okx_pressure:short_carry_premium_watch | 13 | 13 | -0.003216 | 0.307692 | 0.004037 | -0.030560 | 0.000000 | not supported by first labels |
| okx_liquidation:long_liquidation_cascade_watch | 8 | 8 | -0.000119 | 0.375000 | 0.005715 | -0.003874 | 0.000000 | not supported by first labels |

Interpretation:

- `short_liquidation_squeeze_watch` is the strongest current signal family:
  17 covered labels, 0.88 hit rate, and positive mean 15m continuation.
- `long_carry_discount_watch` also has initial support, but the average label
  is much smaller.
- The current short-side carry families are not supported by first labels.
