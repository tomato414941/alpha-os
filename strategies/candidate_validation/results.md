# Candidate Validation Results

Data:

- source: current strategy candidate CSVs
- market data: Hyperliquid public candle snapshots
- output: recent return and volume context for active candidates
- forward label: elapsed monitor samples joined to subsequent Hyperliquid candle returns
- L2 monitor: persistent visible-book imbalance candidates are included as
  pending cross-lane candidates until their forward labels mature

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
| WLD | cross_exchange_funding | 0.48702000 | -0.045714 | 0.144099 | 0.214695 | 143488865.00 | single_source_momentum_context | 26.776315 |
| STABLE | cross_exchange_funding;perp_carry_reversion | 0.03440400 | 0.020981 | 0.032750 | 0.050568 | 36791971.00 | multi_source_momentum_context | 23.735608 |
| MEGA | perp_carry_reversion | 0.04846500 | 0.032356 | 0.101552 | 0.101027 | 28396102.00 | single_source_momentum_context | 23.313251 |
| ONDO | l2_imbalance_monitor | 0.34891000 | -0.008412 | 0.042487 | 0.068016 | 42801976.00 | single_source_momentum_context | 17.965574 |
| XPL | l2_imbalance_monitor | 0.06878500 | -0.015247 | 0.022430 | 0.054225 | 105906498.00 | single_source_context | 17.646195 |
| XLM | l2_imbalance_monitor | 0.20475000 | 0.008372 | 0.014216 | -0.033286 | 35764464.00 | single_source_context | 16.548051 |
| BABY | perp_carry_reversion | 0.01563200 | 0.003982 | 0.018239 | 0.024579 | 104070325.00 | single_source_context | 16.310135 |
| LIT | l2_imbalance_monitor | 1.39920000 | -0.003064 | 0.019825 | -0.047126 | 11228074.00 | single_source_context | 16.297631 |
| SUI | l2_imbalance_monitor | 0.74630000 | -0.003804 | 0.015982 | 0.043338 | 44146290.20 | single_source_context | 16.179549 |
| AERO | attention_market_join;perp_carry_reversion | 0.33075000 | -0.000332 | 0.014415 | 0.038364 | 1093933.00 | multi_source_watch | 11.847928 |

Interpretation:

- `WLD`, `STABLE`, and `MEGA` still have the strongest current realized move
  context among active candidates.
- `ONDO`, `XPL`, `XLM`, `LIT`, and `SUI` are now visible because the L2
  imbalance monitor showed persistent visible-book pressure.
- `AERO` remains structurally interesting because attention and perp carry
  reversion overlap, but its latest short price label is weak.
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
| BTC | 2.6217 | okx_pressure; okx_liquidation; l2_imbalance_monitor | liq_cont15=0.0020; l2_imbalance15=0.0001 | okx_pressure15=-0.0021 |  | mixed evidence; isolate which source is real |
| ALLO | 2.5965 | okx_pressure; okx_liquidation | liq_cont15=0.0198 | okx_pressure15=-0.0078 |  | mixed evidence; isolate which source is real |
| XMR | 2.5530 | hl_candidate_label | hl15=0.0111 |  |  | first labels support follow-up |
| JTO | 2.4579 | okx_pressure; okx_liquidation; l2_imbalance_monitor | liq_cont15=0.0003; l2_imbalance15=0.0125 | okx_pressure15=-0.0010 |  | mixed evidence; isolate which source is real |
| HOME | 2.3356 | okx_pressure; okx_liquidation | okx_pressure15=0.0070 | liq_cont15=-0.0074 |  | mixed evidence; isolate which source is real |
| H | 2.2846 | okx_pressure; okx_liquidation | liq_cont15=0.0131 | okx_pressure15=-0.0005 |  | mixed evidence; isolate which source is real |
| ZORA | 2.2743 | hl_candidate_label | hl15=0.0055 |  |  | first labels support follow-up |
| SOL | 2.1187 | okx_pressure; okx_liquidation; l2_imbalance_monitor | okx_pressure15=0.0031; liq_cont15=0.0017 | l2_imbalance15=-0.0033 |  | mixed evidence; isolate which source is real |
| ONDO | 1.8698 | okx_pressure; okx_liquidation; l2_imbalance_monitor | liq_cont15=0.0020 | okx_pressure15=-0.0029; l2_imbalance15=-0.0046 |  | mixed evidence; isolate which source is real |

Interpretation:

- `WLD` is now the cleanest current follow-up because HL forward label, OKX
  pressure label, and OKX liquidation continuation label all support it.
- `JTO` is the strongest L2-added cross-lane follow-up, but it is mixed because
  OKX pressure is negative.
- `BTC` has a positive L2 label, but it is tiny; it should be a control rather
  than a lead.
- `SOL` and `ONDO` were promising from monitor/cross-lane context, but their
  L2 direction-aware labels are negative.

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
| l2_imbalance:visible_book_imbalance | 23 | 23 | -0.001173 | 0.391304 | 0.012475 | -0.015156 | 0.000000 | not supported by first labels |

Interpretation:

- `short_liquidation_squeeze_watch` is the strongest current signal family:
  17 covered labels, 0.88 hit rate, and positive mean 15m continuation.
- `long_carry_discount_watch` also has initial support, but the average label
  is much smaller.
- `visible_book_imbalance` is now labeled and is not supported as a broad
  family on this first snapshot: negative mean and 0.39 hit rate.
- The current short-side carry families are not supported by first labels.
