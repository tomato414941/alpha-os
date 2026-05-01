# Migration Map

## Purpose

This file maps repository components that may need to be constrained, demoted,
or removed.

It asks:

- which parts are still useful
- which parts are transitional scaffolding
- which parts should eventually be removed, demoted, or absorbed into a
  different layer

## Current Map

### Observation Plane

#### Aligned

- [`observation_adapters.py`](../../src/alpha_os/observation_adapters.py)
  - aligned because it treats `signal-noise` as an observation provider behind a
    contract rather than exposing backend signal names directly
- [`observables.py`](../../src/alpha_os/observables.py)
  - aligned because it makes observables first-class and separate from subject
    identity
- [`signal_client.py`](../../src/alpha_os/signal_client.py)
  - aligned as a thin boundary module to the observation provider

#### Transitional

- [`portfolio_decision.py`](../../src/alpha_os/portfolio_decision.py)
  - transitional in the observation sense because `ObservationSpec` still lives
    inside the portfolio package rather than a dedicated observation namespace
- [`store.py`](../../src/alpha_os/store.py)
  - transitional because observation provenance is still persisted in the same
    general runtime store as evaluation and decision state

#### Remove Or Demote

- any future reintroduction of raw backend `signal_name` into public contracts
  - this should be treated as a regression, not as a supported layer

### Representation Plane

#### Aligned

- [`feature_plane.py`](../../src/alpha_os/feature_plane.py)
  - aligned because it makes subject-level shared representation the primary
    unit of computation
- [`evaluation_generation.py`](../../src/alpha_os/evaluation_generation.py)
  - aligned where it uses feature planes and batch family execution rather than
    one-fetch-per-signal logic

#### Transitional

- [`evaluation_inputs.py`](../../src/alpha_os/evaluation_inputs.py)
  - transitional because it still materializes many row-like evaluation objects
    as a primary runtime interface
- [`evaluation_runtime.py`](../../src/alpha_os/evaluation_runtime.py)
  - transitional because it persists per-evaluation records as first-class
    runtime truth rather than treating them as one possible byproduct of discovery

#### Remove Or Demote

- one-row-at-a-time evaluation persistence as the dominant execution shape
  - batch persistence may remain as a bridge, but row-first evaluation should
    not define the long-run architecture

### Signal Discovery Plane

#### Aligned

- [`signal_compiler.py`](../../src/alpha_os/signal_compiler.py)
  - aligned because it groups execution by family structure rather than by
    signal identity
- [`signal_registry.py`](../../src/alpha_os/signal_registry.py)
  - partially aligned where it distinguishes specification from executable
    binding and tracks required observables

#### Transitional

- [`signal_registry.py`](../../src/alpha_os/signal_registry.py)
  - still transitional overall because executable signals remain durable
    runtime entities
- [`targets.py`](../../src/alpha_os/targets.py)
  - transitional because targets are still closer to bounded evaluation
    contracts than to a broader discovery language

#### Remove Or Demote

- executable signal IDs as a long-run source-of-truth abstraction
  - they may remain as runtime cache keys or audit keys
  - they should not remain the main conceptual unit of large-scale discovery

### Selection Plane

#### Aligned

- [`screening.py`](../../src/alpha_os/screening.py)
  - aligned as the current first-class `ScreeningResult` and
    `ScreeningPolicy` artifact layer
- [`signal_discovery_screening_service.py`](../../src/alpha_os/signal_discovery_screening_service.py)
  - aligned where it turns discovery outputs into persisted screening
    artifacts

#### Transitional

- [`pre_screening.py`](../../src/alpha_os/pre_screening.py),
  [`probe_screening.py`](../../src/alpha_os/probe_screening.py), and
  [`survivor_screening.py`](../../src/alpha_os/survivor_screening.py)
  - transitional because staged screening exists, but the policy/reporting
    surface is still being tightened
- [`metrics_service.py`](../../src/alpha_os/metrics_service.py)
  - transitional because it provides useful scoring, but not yet a proper
    staged screening layer
- [`meta_aggregation_service.py`](../../src/alpha_os/meta_aggregation_service.py)
  - transitional because it aggregates beliefs, but only after broad
    evaluation, not after first-class screening
- [`meta_metrics_service.py`](../../src/alpha_os/meta_metrics_service.py)
  - transitional for the same reason
- [`scoring.py`](../../src/alpha_os/scoring.py)
  - transitional because scoring exists, but not yet as part of a dedicated
    cheap-screen versus deep-evaluation split

#### Remove Or Demote

- broad full-run validation as the first filter
  - this should become a survivor-stage activity, not the default selection
    mechanism
- any tendency for cheap pre-screen to decide final stability, final novelty, or
  portfolio value
  - cheap pre-screen should reject obvious losers only

### Compression Plane

#### Aligned

- [`compression.py`](../../src/alpha_os/compression.py)
  - aligned as the current first-class `CompressedBelief` artifact layer
- [`signal_discovery_compression_service.py`](../../src/alpha_os/signal_discovery_compression_service.py)
  - aligned where it persists compression outputs from screening survivors

#### Transitional

- [`belief_synthesis.py`](../../src/alpha_os/belief_synthesis.py)
  - transitional because it supports current compression, but remains heuristic
    rather than a final factor model
- [`meta_aggregation_service.py`](../../src/alpha_os/meta_aggregation_service.py)
  - transitional as an older aggregation path, not the main compressed-belief
    artifact owner
- [`portfolio_decision_inputs.py`](../../src/alpha_os/portfolio_decision_inputs.py)
  - transitional because uncertainty and dependence inputs already hint at
    compression-aware thinking, but they still consume broad upstream results

#### Remove Or Demote

- direct use of large raw signal populations in portfolio decision
  - the target is compressed belief, not broad raw signal sets

### Decision Plane

#### Aligned

- [`portfolio_decision.py`](../../src/alpha_os/portfolio_decision.py)
  - aligned because it defines the decision contract in portfolio-facing terms
- [`portfolio_decision_inputs.py`](../../src/alpha_os/portfolio_decision_inputs.py)
  - aligned where it separates observed inputs from assumptions
- [`portfolio_execution_policy.py`](../../src/alpha_os/portfolio_execution_policy.py)
  - aligned where it keeps execution policy separate from belief and sizing
- [`portfolio_decision_service.py`](../../src/alpha_os/portfolio_decision_service.py)
  - aligned where it acts as a composition layer rather than a place that
    invents portfolio meaning
- [`decision_backtest.py`](../../src/alpha_os/decision_backtest.py)
  - aligned where it treats replay as portfolio-state evolution

#### Transitional

- rule-based policy support inside
  [`portfolio_sizing_policy.py`](../../src/alpha_os/portfolio_sizing_policy.py)
  - transitional because the long-run system should center on compressed belief
    plus constrained decision, not on static rule transforms
- current uncertainty inputs
  - transitional because `estimation`, `model`, and `structural` uncertainty
    are not yet deeply integrated into a compressed-belief pipeline

#### Remove Or Demote

- any future tendency to use the decision layer as a substitute for missing
  upstream screening or compression

### Evaluation Plane

#### Aligned

- [`validation_engine.py`](../../src/alpha_os/validation_engine.py)
  - aligned where it distinguishes replay and outcome measurement from raw
    generation
- [`validation_service.py`](../../src/alpha_os/validation_service.py)
  - aligned where it scopes validation around subject sets and portfolio
    outcomes

#### Transitional

- [`validation_spec.py`](../../src/alpha_os/validation_spec.py)
  - transitional because it still reflects broad replay-oriented validation
    rather than a multi-stage survivor evaluation pipeline
- [`validation_service.py`](../../src/alpha_os/validation_service.py)
  - transitional because validation still runs too close to a full broad pass
    rather than a staged selection process

#### Remove Or Demote

- winner-picking interpretations of validation output
  - validation should describe failure surfaces and robustness, not only ranking

### Control Surface

#### Aligned

- workflow-first CLI in [`cli/`](../../src/alpha_os/cli/)
  - aligned because public commands now prefer workflow units over small legacy
    primitives

#### Transitional

- [`cli/`](../../src/alpha_os/cli/)
  - transitional because the package has runtime/evaluation/research/internal
    handler groups, but some parser and command implementation still lives in
    the legacy internal module
- [`cli_output.py`](../../src/alpha_os/cli_output.py)
  - transitional because it still exposes runtime artifacts more than higher
    level discovery and compression summaries
- [`config.py`](../../src/alpha_os/config.py)
  - transitional because bounded defaults still exist across runtime paths
- [`store.py`](../../src/alpha_os/store.py)
  - transitional because one store still carries multiple conceptual planes

#### Remove Or Demote

- debug commands that could drift back into public runtime truth
