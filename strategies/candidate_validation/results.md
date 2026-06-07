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
uv run python -m strategies.candidate_validation.current_followup_repeat_observations
uv run python -m strategies.candidate_validation.current_followup_repeat_forward_labels
uv run python -m strategies.candidate_validation.current_followup_venue_coverage
uv run python -m strategies.candidate_validation.current_followup_okx_execution_context
uv run python -m strategies.candidate_validation.current_followup_okx_repeat_observations
uv run python -m strategies.candidate_validation.current_followup_okx_repeat_forward_labels
uv run python -m strategies.candidate_validation.current_followup_repeat_history
uv run python -m strategies.candidate_validation.current_followup_repeat_history_labels
uv run python -m strategies.candidate_validation.current_followup_repeat_history_summary
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

## Current Follow-Up Repeat Observations

| asset | source | source action | dir | priority | mark | funding ann | spread bps | depth 10bps USD | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| WLD | okx_pressure | long_carry_discount_watch | 1 | 10.0571 | 0.48648000 | -0.214318 | 5.3436 | 22766 | ready_for_label |
| WLD | liquidation | short_liquidation_squeeze_watch | 1 | 10.0571 | 0.48648000 | -0.214318 | 5.3436 | 22766 | ready_for_label |
| ETH | okx_pressure | long_carry_discount_watch | 1 | 4.5510 | 1633.00000000 | 0.065543 | 0.6123 | 12184113 | ready_for_label |
| ETH | liquidation | short_liquidation_squeeze_watch | 1 | 4.5510 | 1633.00000000 | 0.065543 | 0.6123 | 12184113 | ready_for_label |
| ETH | l2_imbalance | visible_book_imbalance | 1 | 4.5510 | 1633.00000000 | 0.065543 | 0.6123 | 12184113 | ready_for_label |
| ONDO | liquidation | short_liquidation_squeeze_watch | 1 | 3.6106 | 0.34858000 | 0.109500 | 0.8600 | 35880 | ready_for_label |
| ONDO | sector_rotation | sector_momentum_watch | 1 | 3.6106 | 0.34858000 | 0.109500 | 0.8600 | 35880 | ready_for_label |
| XPL | l2_imbalance | visible_book_imbalance | 1 | 3.4493 | 0.06900700 | 0.109500 | 3.3322 | 5094 | ready_for_label |
| XPL | sector_rotation | sector_momentum_watch | 1 | 3.4493 | 0.06900700 | 0.109500 | 3.3322 | 5094 | ready_for_label |

Interpretation:

- Fresh repeat observations are now source-specific. `ONDO/liquidation` and
  `ONDO/sector_rotation` are separate observations, not one blended candidate.
- 23 rows are ready for 15m/1h labels.
- `WLD` has no reusable `hl_candidate` direction in the current label format, so
  only OKX pressure and liquidation are ready for directional repeat labels.

## Current Follow-Up Repeat Forward Labels

The fresh repeat observations are currently `pending_15m`. Rerun:

```bash
uv run python -m strategies.candidate_validation.current_followup_repeat_forward_labels
```

after the 15m and 1h horizons mature.

## Current Follow-Up Venue Coverage

| asset | priority | source | HL | OKX | Binance | venues | action | reason |
| --- | ---: | --- | --- | --- | --- | ---: | --- | --- |
| WLD | 10.0571 | hl_candidate;okx_pressure;liquidation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| ETH | 4.5510 | okx_pressure;liquidation;l2_imbalance | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| MEGA | 3.8916 | hl_candidate | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| BTC | 3.6217 | liquidation;l2_imbalance | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| ONDO | 3.6106 | liquidation;sector_rotation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| XPL | 3.4493 | l2_imbalance;sector_rotation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| PUMP | 2.9792 | liquidation;sector_rotation | True | True | False | 2 | multi_venue_followup | candidate can be observed or routed on multiple perp venues |
| PEPE | 3.7269 | okx_pressure;liquidation | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |
| ALLO | 3.0965 | liquidation | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |
| HOME | 2.8356 | okx_pressure | False | True | False | 1 | okx_only_followup | candidate is missing from Hyperliquid but exists on OKX USDT swap |

Interpretation:

- Hyperliquid-only execution context was too narrow.
- `PEPE`, `ALLO`, `HOME`, `H`, `LAB`, and `BEAT` are not HL candidates, but
  they are OKX USDT swap candidates.
- Binance futures metadata returned `451` from this environment, so Binance
  coverage is currently unavailable rather than proven absent.

## Current Follow-Up OKX Execution Context

| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| WLD | WLD-USDT-SWAP | hl_candidate;okx_pressure;liquidation | 10.0571 | -0.439141 | 1062380848 | 2.0536 | 29416 | 0.033995 | okx_context_ok |
| ETH | ETH-USDT-SWAP | okx_pressure;liquidation;l2_imbalance | 4.5510 | -1.143435 | 4294945 | 0.0613 | 1293112 | 0.000773 | okx_context_ok |
| PEPE | PEPE-USDT-SWAP | okx_pressure;liquidation | 3.7269 | 0.529954 | 43756031000000 | 3.5913 | 175358 | 0.005703 | okx_context_ok |
| ONDO | ONDO-USDT-SWAP | liquidation;sector_rotation | 3.6106 | -0.135784 | 115289430 | 2.8939 | 24572 | 0.040696 | okx_context_ok |
| JTO | JTO-USDT-SWAP | liquidation;l2_imbalance | 3.4579 | 0.438000 | 97435983 | 1.5769 | 7434 | 0.134518 | okx_context_ok |
| XPL | XPL-USDT-SWAP | l2_imbalance;sector_rotation | 3.4493 | -0.224184 | 387997670 | 1.4475 | 12698 | 0.078752 | okx_context_ok |
| ALLO | ALLO-USDT-SWAP | liquidation | 3.0965 | 0.239926 | 969418890 | 0.3224 | 5722 | 0.174754 | okx_context_ok |
| PUMP | PUMP-USDT-SWAP | liquidation;sector_rotation | 2.9792 | 0.438000 | 24647151000 | 6.6203 | 22979 | 0.043518 | okx_context_ok |
| H | H-USDT-SWAP | liquidation | 2.7846 | 0.438000 | 144645250 | 0.1245 | 5428 | 0.184221 | okx_context_ok |

Interpretation:

- OKX keeps several HL-missing candidates alive: `PEPE`, `ALLO`, and `H` are
  not dead simply because Hyperliquid lacks them.
- `HOME` exists on OKX but fails the rough 1k visible-depth check in this
  snapshot, so it needs smaller sizing or a different venue/timing.
- This is still public book context only; account fees and fill quality are not
  included.

## Current Follow-Up OKX Repeat Observations

| asset | source | source action | dir | priority | inst | last | funding ann | spread bps | depth 10bps USD | status |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| WLD | okx_pressure | long_carry_discount_watch | 1 | 10.0571 | WLD-USDT-SWAP | 0.48710000 | -0.439141 | 2.0536 | 29416 | ready_for_label |
| WLD | liquidation | short_liquidation_squeeze_watch | 1 | 10.0571 | WLD-USDT-SWAP | 0.48710000 | -0.439141 | 2.0536 | 29416 | ready_for_label |
| ETH | okx_pressure | long_carry_discount_watch | 1 | 4.5510 | ETH-USDT-SWAP | 1632.19000000 | -1.143435 | 0.0613 | 1293112 | ready_for_label |
| PEPE | okx_pressure | long_carry_watch | 1 | 3.7269 | PEPE-USDT-SWAP | 0.00000278 | 0.529954 | 3.5913 | 175358 | ready_for_label |
| PEPE | liquidation | short_liquidation_squeeze_watch | 1 | 3.7269 | PEPE-USDT-SWAP | 0.00000278 | 0.529954 | 3.5913 | 175358 | ready_for_label |
| ONDO | liquidation | short_liquidation_squeeze_watch | 1 | 3.6106 | ONDO-USDT-SWAP | 0.34550000 | -0.135784 | 2.8939 | 24572 | ready_for_label |
| ONDO | sector_rotation | sector_momentum_watch | 1 | 3.6106 | ONDO-USDT-SWAP | 0.34550000 | -0.135784 | 2.8939 | 24572 | ready_for_label |
| ALLO | liquidation | short_liquidation_squeeze_watch | 1 | 3.0965 | ALLO-USDT-SWAP | 0.31018000 | 0.239926 | 0.3224 | 5722 | ready_for_label |
| H | liquidation | short_liquidation_squeeze_watch | 1 | 2.7846 | H-USDT-SWAP | 0.80359000 | 0.438000 | 0.1245 | 5428 | ready_for_label |

Interpretation:

- OKX repeat observations add `PEPE`, `ALLO`, and `H`, which Hyperliquid repeat
  observations could not cover.
- 28 OKX source-specific rows are ready for 15m/1h labels.
- The OKX repeat labels are currently `pending_15m`; rerun
  `current_followup_okx_repeat_forward_labels` after maturity.

## Current Follow-Up Repeat History

The repeat history preserves source-specific observations across runs.

- total rows: `106`
- ready rows: `102`
- by venue: `HL=46; OKX=56`
- by source: `hl_candidate=2; l2_imbalance=16; liquidation=42; okx_pressure=30; sector_rotation=12`

Interpretation:

- Current observation files can be regenerated, so they are not enough for
  repeated alpha checks.
- This history keeps HL and OKX repeat samples in one place without blending
  source meanings.

## Current Follow-Up Repeat History Labels

| venue | asset | source | action | dir | priority | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| OKX | JTO | liquidation | long_liquidation_cascade_watch | -1 | 3.4579 | -0.020572 | 0.020572 |  |  | labeled_15m_pending_1h |
| OKX | JTO | liquidation | long_liquidation_cascade_watch | -1 | 3.4579 | -0.011865 | 0.011865 |  |  | labeled_15m_pending_1h |
| HL | LTC | liquidation | long_liquidation_cascade_watch | -1 | 3.2959 | -0.004115 | 0.004115 |  |  | labeled_15m_pending_1h |
| OKX | XLM | okx_pressure | long_carry_discount_watch | 1 | 2.9178 | 0.003891 | 0.003891 |  |  | labeled_15m_pending_1h |
| OKX | XLM | l2_imbalance | visible_book_imbalance | 1 | 2.9178 | 0.003891 | 0.003891 |  |  | labeled_15m_pending_1h |
| HL | XLM | okx_pressure | long_carry_discount_watch | 1 | 2.9178 | 0.003793 | 0.003793 |  |  | labeled_15m_pending_1h |
| HL | XLM | l2_imbalance | visible_book_imbalance | 1 | 2.9178 | 0.003793 | 0.003793 |  |  | labeled_15m_pending_1h |
| OKX | ALLO | liquidation | short_liquidation_squeeze_watch | 1 | 3.0965 | 0.003645 | 0.003645 |  |  | labeled_15m_pending_1h |
| OKX | LTC | liquidation | long_liquidation_cascade_watch | -1 | 3.2959 | -0.003331 | 0.003331 |  |  | labeled_15m_pending_1h |
| OKX | TON | liquidation | short_liquidation_squeeze_watch | 1 | 2.1872 | 0.002904 | 0.002904 |  |  | labeled_15m_pending_1h |
| OKX | TON | okx_pressure | short_carry_watch | -1 | 2.1872 | -0.002316 | 0.002316 |  |  | labeled_15m_pending_1h |
| OKX | PUMP | liquidation | short_liquidation_squeeze_watch | 1 | 2.9792 | 0.001993 | 0.001993 |  |  | labeled_15m_pending_1h |

Interpretation:

- Two repeat batches are stored and all 102 ready rows now have 15m labels.
- `JTO/liquidation` is the strongest repeat label and now has two positive
  OKX rows.
- `LTC/liquidation` is the cleanest multi-venue repeat group after the second
  batch matured.
- `XLM`, `PUMP`, and `XRP` were positive in the first batch but became mixed
  after the second batch, so they should not be promoted as clean repeats.
- `WLD` remains weak in repeat history despite the earlier cross-lane score.

## Current Follow-Up Repeat History Summary

| group type | group | labeled | pending | hit 15m | mean dir15 | action |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| asset_source | JTO/liquidation | 2 | 0 | 1.000 | 0.016218 | repeat_priority |
| venue_asset_source | OKX/JTO/liquidation | 2 | 0 | 1.000 | 0.016218 | repeat_priority |
| venue_asset_source | HL/LTC/liquidation | 2 | 0 | 1.000 | 0.002912 | repeat_priority |
| asset_source | LTC/liquidation | 4 | 0 | 1.000 | 0.002705 | repeat_priority |
| venue_asset_source | OKX/LTC/liquidation | 2 | 0 | 1.000 | 0.002497 | repeat_priority |

Interpretation:

- `JTO/liquidation` is the strongest clean repeat but is OKX-only so far.
- `LTC/liquidation` is lower magnitude but more structurally interesting because
  it has positive labeled rows on both HL and OKX.
- `XLM` has a direct conflict: repeat price labels are now mixed and chain TVL
  flow is negative.
- The next useful work is to collect another liquidation-specific repeat batch
  for `JTO` and `LTC`, then add rough costs/funding/slippage.
