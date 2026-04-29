# Migration Map

## Purpose

This file maps the current repository onto the greenfield target described in
[`signal-discovery-system.md`](./signal-discovery-system.md).

It does not ask whether every current module is useful today.

It asks:

- which parts already fit the target
- which parts are transitional scaffolding
- which parts should eventually be removed, demoted, or absorbed into a
  different layer

## Classification Rules

### Aligned

An aligned component already matches the intended greenfield shape closely
enough that it should be strengthened rather than replaced.

### Transitional

A transitional component is useful now, but it reflects the migration path more
than the target architecture.

These components should usually be:

- constrained
- isolated
- prevented from becoming more central

### Remove Or Demote

A remove-or-demote component should not remain part of long-run runtime truth.

It may survive as:

- a debug path
- a compatibility layer
- a one-time migration bridge

But it should not be expanded.

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
    contracts than to a full greenfield discovery language

#### Remove Or Demote

- executable signal IDs as a long-run source-of-truth abstraction
  - they may remain as runtime cache keys or audit keys
  - they should not remain the main conceptual unit of large-scale discovery

### Selection Plane

#### Aligned

- no current module fully owns this plane yet

#### Transitional

- cheap pre-screen inside
  [`evaluation_generation.py`](../../src/alpha_os/evaluation_generation.py)
  - transitional because it is useful and correctly front-loads cheap rejection,
    but it is still shaped by current batch backfill mechanics rather than by a
    first-class greenfield pre-screen contract
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

- no current module fully owns this plane yet

#### Transitional

- [`meta_aggregation_service.py`](../../src/alpha_os/meta_aggregation_service.py)
  - transitional because it is the closest current approximation to belief
    compression, but it still operates as simple aggregation rather than
    explicit redundancy reduction or factor compression
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
- [`portfolio_decision_policy.py`](../../src/alpha_os/portfolio_decision_policy.py)
  - aligned where it treats decision as a function of belief, state, cost,
    dependence, and uncertainty
- [`portfolio_decision_service.py`](../../src/alpha_os/portfolio_decision_service.py)
  - aligned where it acts as a composition layer rather than a place that
    invents portfolio meaning
- [`decision_backtest.py`](../../src/alpha_os/decision_backtest.py)
  - aligned where it treats replay as portfolio-state evolution

#### Transitional

- `rule` policy support inside
  [`portfolio_decision_policy.py`](../../src/alpha_os/portfolio_decision_policy.py)
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

- workflow-first CLI in [`cli.py`](../../src/alpha_os/cli.py)
  - aligned because public commands now prefer workflow units over small legacy
    primitives

#### Transitional

- [`cli.py`](../../src/alpha_os/cli.py)
  - still transitional because workflows are still shaped by backfill and
    broad-evaluation concepts rather than by discovery, screening, and
    compression concepts
- [`cli_output.py`](../../src/alpha_os/cli_output.py)
  - transitional because it still exposes runtime artifacts more than higher
    level discovery and compression summaries
- [`config.py`](../../src/alpha_os/config.py)
  - transitional because bounded defaults still exist even though the design
    target is larger and more explicit
- [`store.py`](../../src/alpha_os/store.py)
  - transitional because one store still carries multiple conceptual planes

#### Remove Or Demote

- debug commands that could drift back into public runtime truth

## Near-Term Direction

To move closer to the greenfield target, the next architectural additions
should be:

1. a first-class `SignalDiscovery`
2. a first-class `ScreeningResult`
3. a first-class compression layer that turns many surviving signals into a
   smaller belief surface

The next architectural reductions should be:

1. treating executable signals as durable conceptual units
2. treating broad backfill as the main shape of large-scale discovery
3. treating full validation as the default first pass

## Practical Rule

When deciding whether to add or change a module, ask:

- does this strengthen observation, representation, discovery, selection,
  compression, or decision as separate planes

and:

- does this reduce the centrality of executable-signal runtime truth

If the answer is no, the change is probably transitional at best.
