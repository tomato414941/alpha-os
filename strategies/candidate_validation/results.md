# Candidate Validation Results

Data:

- source: current strategy candidate CSVs across research lanes
- market data: Hyperliquid public candle snapshots
- forward labels: elapsed monitor/category samples joined to subsequent returns
- purpose: cross-lane triage, not deployable strategy ranking

Run:

```bash
uv run python -m strategies.candidate_validation.current_hl_candidate_return_context
uv run python -m strategies.candidate_validation.current_hl_signal_forward_labels
uv run python -m strategies.candidate_validation.current_cross_lane_candidate_review
uv run python -m strategies.candidate_validation.current_signal_family_review
uv run python -m strategies.candidate_validation.current_source_conflict_review
uv run python -m strategies.candidate_validation.current_followup_queue
uv run python -m strategies.candidate_validation.current_followup_execution_context
```

This is not a causal alpha test. It keeps candidates connected to realized
market behavior so screens do not stay detached from price, volume, and
short-horizon labels.

## Current HL Candidate Return Context

| symbol | sources | 1h | 4h | 24h | action | score |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| MEGA | perp_carry_reversion | 0.051728 | 0.132056 | 0.148115 | single_source_momentum_context | 26.775569 |
| ONDO | l2_imbalance_monitor;sector_rotation | -0.005770 | 0.027165 | 0.053567 | multi_source_watch | 21.935217 |
| STABLE | cross_exchange_funding;perp_carry_reversion | -0.005979 | 0.016281 | 0.047003 | multi_source_watch | 21.411987 |
| XPL | l2_imbalance_monitor;sector_rotation | 0.003730 | 0.008299 | 0.058187 | multi_source_watch | 20.787957 |
| WLD | cross_exchange_funding | -0.001421 | 0.095618 | 0.204984 | single_source_momentum_context | 19.922947 |
| XMR | perp_carry_reversion;sector_rotation | 0.017081 | 0.039545 | 0.039889 | multi_source_momentum_context | 13.712548 |

Interpretation:

- `sector_rotation` now contributes to candidate context instead of staying in
  its own silo.
- `ONDO` and `XPL` are now multi-source candidates because sector rotation
  overlaps with existing L2/perp context.
- `XMR` is multi-source and has recent positive 1h/4h movement, but its sector
  forward label is negative, so the source conflict needs isolation.

## Current HL Signal Forward Labels

| source | action | asset | obs | cov15 | cov1h | mean 15m | mean 1h | hit15 | hit1h |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| okx_hl_current | paper_24h_monitor | WLD | 10 | 10 | 10 | 0.019682 | 0.072090 | 1.000000 | 1.000000 |
| perp_carry_reversion | long_carry_reversion_watch | MEGA | 6 | 6 | 0 | 0.017831 |  | 1.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | IP | 6 | 6 | 0 | 0.015990 |  | 1.000000 |  |
| perp_carry_reversion | short_carry_reversion_watch | XMR | 6 | 6 | 0 | 0.011059 |  | 1.000000 |  |
| perp_carry_reversion | long_carry_reversion_watch | ZORA | 6 | 6 | 0 | 0.005486 |  | 1.000000 |  |

Interpretation:

- `WLD`, `MEGA`, `IP`, and `XMR` still have positive elapsed 15m forward price
  labels from existing monitor samples.
- This section does not include the new sector labels; those appear in the
  cross-lane and family reviews below.

## Current Cross-Lane Candidate Review

| asset | score | lanes | positive labels | negative labels | note |
| --- | ---: | --- | --- | --- | --- |
| WLD | 7.0571 | hl_candidate_label; okx_pressure; okx_liquidation | hl15=0.0197; okx_pressure15=0.0247; liq_cont15=0.0273 |  | first labels support follow-up |
| XMR | 3.1004 | hl_candidate_label; sector_rotation | hl15=0.0111 | sector15=-0.0005:Privacy | mixed evidence; isolate which source is real |
| MEGA | 2.8916 | hl_candidate_label | hl15=0.0178 |  | first labels support follow-up |
| ONDO | 2.6106 | okx_pressure; okx_liquidation; l2_imbalance_monitor; sector_rotation | liq_cont15=0.0020; sector15=0.0029:Binance Alpha Spotlight | okx_pressure15=-0.0029; l2_imbalance15=-0.0046 | mixed evidence; isolate which source is real |
| JTO | 2.4579 | okx_pressure; okx_liquidation; l2_imbalance_monitor | liq_cont15=0.0003; l2_imbalance15=0.0125 | okx_pressure15=-0.0010 | mixed evidence; isolate which source is real |
| XPL | 2.4493 | okx_pressure; l2_imbalance_monitor; sector_rotation | l2_imbalance15=0.0030; sector15=0.0035:Echo Launchpad | okx_pressure15=-0.0037 | mixed evidence; isolate which source is real |
| PUMP | 1.9792 | okx_pressure; okx_liquidation; sector_rotation | liq_cont15=0.0020; sector15=0.0027:Launchpad | okx_pressure15=-0.0020 | mixed evidence; isolate which source is real |

Interpretation:

- `WLD` remains the cleanest multi-label follow-up.
- `ONDO`, `XPL`, and `PUMP` are now more interesting because sector rotation
  supports them, but they are not clean: other short labels conflict.
- The useful next step is source isolation: determine whether sector rotation,
  liquidation continuation, L2 imbalance, or OKX pressure is carrying the edge.

## Current Signal Family Review

| family | obs | cov15 | mean15 | hit15 | max15 | min15 | score | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| okx_liquidation:short_liquidation_squeeze_watch | 17 | 17 | 0.004270 | 0.882353 | 0.027309 | -0.009277 | 1.387726 | supported by first labels |
| okx_pressure:long_carry_discount_watch | 32 | 32 | 0.001608 | 0.750000 | 0.024699 | -0.008581 | 0.401907 | supported by first labels |
| sector_rotation:sector_momentum_watch | 13 | 13 | 0.001082 | 0.692308 | 0.003547 | -0.002919 | 0.135227 | supported by first labels |
| l2_imbalance:visible_book_imbalance | 23 | 23 | -0.001173 | 0.391304 | 0.012475 | -0.015156 | 0.000000 | not supported by first labels |

Interpretation:

- `short_liquidation_squeeze_watch` remains the strongest current family.
- `sector_momentum_watch` is now a supported family on the first tradable label
  sample: 13 covered labels, 0.69 hit rate, positive mean 15m.
- `visible_book_imbalance` is still not supported as a broad family despite
  some strong individual names.

## Current Source Conflict Review

| asset | score | positives | negatives | action | next test |
| --- | ---: | --- | --- | --- | --- |
| XMR | 3.1004 | hl_candidate | sector_rotation | separate_carry_from_sector | repeat the original candidate family and keep unrelated negative sources out of the decision |
| IP | 2.8166 | hl_candidate | okx_pressure | isolate_positive_source | repeat the original candidate family and keep unrelated negative sources out of the decision |
| BTC | 2.6217 | liquidation;l2_imbalance | okx_pressure | repeat_liquidation_not_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| ONDO | 2.6106 | liquidation;sector_rotation | okx_pressure;l2_imbalance | separate_sector_from_l2 | repeat sector labels with category membership and costs before mixing with other sources |
| JTO | 2.4579 | liquidation;l2_imbalance | okx_pressure | repeat_liquidation_not_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| XPL | 2.4493 | l2_imbalance;sector_rotation | okx_pressure | repeat_l2_not_pressure | repeat sector labels with category membership and costs before mixing with other sources |
| PUMP | 1.9792 | liquidation;sector_rotation | okx_pressure | repeat_liquidation_not_pressure | repeat sector labels with category membership and costs before mixing with other sources |

Interpretation:

- Mixed evidence is not a reason to average everything together.
- `ONDO`, `XPL`, and `PUMP` should be retested as sector/liquidation/L2
  candidates separately from OKX pressure.
- `XMR` should be treated as carry/reversion evidence, not as a sector
  momentum continuation candidate.

## Current Follow-Up Queue

| priority | asset | source | type | action | evidence | next test |
| ---: | --- | --- | --- | --- | --- | --- |
| 10.0571 | WLD | hl_candidate;okx_pressure;liquidation | clean_candidate_repeat | repeat_supported_candidate | hl15=0.0197;okx_pressure15=0.0247;liq_cont15=0.0273 | repeat the same labels on fresh samples and add rough costs |
| 4.5510 | ETH | okx_pressure;liquidation;l2_imbalance | clean_candidate_repeat | repeat_supported_candidate | okx_pressure15=0.0007;liq_cont15=0.0011;l2_imbalance15=0.0010 | repeat the same labels on fresh samples and add rough costs |
| 3.8916 | MEGA | hl_candidate | clean_candidate_repeat | repeat_supported_candidate | hl15=0.0178 | repeat the same labels on fresh samples and add rough costs |
| 3.7269 | PEPE | okx_pressure;liquidation | clean_candidate_repeat | repeat_supported_candidate | okx_pressure15=0.0040;liq_cont15=0.0033 | repeat the same labels on fresh samples and add rough costs |
| 3.6217 | BTC | liquidation;l2_imbalance | source_isolation | repeat_liquidation_not_pressure | positive=liquidation;l2_imbalance;negative=okx_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| 3.6106 | ONDO | liquidation;sector_rotation | source_isolation | separate_sector_from_l2 | positive=liquidation;sector_rotation;negative=okx_pressure;l2_imbalance | repeat sector labels with category membership and costs before mixing with other sources |
| 3.6004 | XMR | hl_candidate | source_isolation | separate_carry_from_sector | positive=hl_candidate;negative=sector_rotation | repeat the original candidate family and keep unrelated negative sources out of the decision |
| 3.4579 | JTO | liquidation;l2_imbalance | source_isolation | repeat_liquidation_not_pressure | positive=liquidation;l2_imbalance;negative=okx_pressure | repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test |
| 3.4493 | XPL | l2_imbalance;sector_rotation | source_isolation | repeat_l2_not_pressure | positive=l2_imbalance;sector_rotation;negative=okx_pressure | repeat sector labels with category membership and costs before mixing with other sources |

Interpretation:

- The project now has a concrete repeat queue instead of a pile of unrelated
  screens.
- `WLD` is the highest-priority clean repeat.
- `ONDO`, `XPL`, and `PUMP` should be tested source-by-source instead of being
  averaged into one generic candidate.

## Current Follow-Up Execution Context

| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| WLD | hl_candidate;okx_pressure;liquidation | 10.0571 | 0.109500 | 64954774 | 5.3436 | 22766 | 0.043925 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ETH | okx_pressure;liquidation;l2_imbalance | 4.5510 | 0.083950 | 422095068 | 0.6123 | 12184113 | 0.000082 | tradable_context_ok | public venue context does not obviously block a small repeat |
| BTC | liquidation;l2_imbalance | 3.6217 | 0.029200 | 240430043 | 0.1607 | 2835549 | 0.000353 | tradable_context_ok | public venue context does not obviously block a small repeat |
| ONDO | liquidation;sector_rotation | 3.6106 | 0.237250 | 13424904 | 0.8600 | 35880 | 0.027870 | tradable_context_ok | public venue context does not obviously block a small repeat |
| XPL | l2_imbalance;sector_rotation | 3.4493 | 0.116800 | 42106313 | 3.3322 | 5094 | 0.196324 | tradable_context_ok | public venue context does not obviously block a small repeat |
| PUMP | liquidation;sector_rotation | 2.9792 | 0.032850 | 151044657 | 6.6203 | 37064 | 0.026981 | tradable_context_ok | public venue context does not obviously block a small repeat |

Interpretation:

- `WLD`, `ETH`, `BTC`, `ONDO`, `XPL`, and `PUMP` are not obviously blocked for
  small repeat observations on public Hyperliquid context.
- `XPL` is tradable but shallow enough that even a 1k repeat uses a noticeable
  share of visible 10 bps depth.
- Some queue names are not current Hyperliquid perp symbols, so venue-specific
  validation is required before treating them as executable candidates.
